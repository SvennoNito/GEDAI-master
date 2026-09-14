function varargout = gedai_band_engine(action, varargin)
%GEDAI_BAND_ENGINE  One GEDAI band pass, delivered as a sequence of output segments.
%
%   state = gedai_band_engine('begin', src, srate, chanlocs, artifact_threshold_type, ...
%               epoch_size, refCOV, optimization_type, parallel, signal_type, ...
%               minThreshold, maxThreshold, smoothing_window_seconds, ...
%               percentile_threshold, rank_truncation, opts)
%   while ~state.done
%       [state, seg, first, last, artifacts] = gedai_band_engine('step', state);
%       % seg holds cleaned samples first..last, in the working precision
%   end
%   [SENSAI_score, artifact_threshold_out, ENOVA] = gedai_band_engine('finish', state);
%
%   This is GEDAI_per_band's computation, reorganised around memory. The matrix
%   version took the band as a matrix, returned the cleaned band as a matrix,
%   and in between held several more band-sized arrays: the band cast to the
%   working precision, a reflect-padded copy, a half-epoch-shifted copy, a
%   cleaned (and on the legacy path an artifact) array per epoch grid, their
%   overlap-added sum, a truncated copy of that and, for ENOVA, cleaned +
%   artifacts. For a whole-night band at 256 channels that was the peak of the
%   pipeline.
%
%   Here the band arrives as a source (gedai_source). A 'wavelet' source is
%   reconstructed once, in pieces, straight into the working precision; a
%   'matrix' source is read in place. The streaming path cleans both epoch
%   grids side by side a block at a time, and each step returns the finished
%   stretch of output - grid 1 plus grid 2 wherever they overlap - for the
%   caller to add straight into its accumulator. The legacy path, whose bands
%   have few long epochs, cleans the whole band in 'begin' and hands the result
%   out in slices. The arithmetic and its order are unchanged, including every
%   choice that can move a floating-point result: the same epoch blocks go to
%   the same kind of worker, the same batched solves see the same number of
%   epochs, ENOVA's var() sees the same column groupings, and the two grids are
%   summed once per sample exactly as the overlap-add did. The output is
%   bit-identical to the matrix version, which GEDAI_per_band still provides
%   on top of this engine.
%
%   The inputs are those of GEDAI_per_band, with the band replaced by a source;
%   see GEDAI_per_band for their meaning. Additional opts field:
%     .legacy_artifacts  return artifacts on the legacy path even when
%                        want_artifacts is false (default true, which is what
%                        GEDAI_per_band always did). GEDAI.m sets false.
%   state.cls is the working precision; state.cast_back is true when the band
%   ran in single although its input was double.
%
%   Two implementations, chosen in 'begin' exactly as GEDAI_per_band chose:
%     stream  global threshold (smoothing_window_seconds = Inf), 'parabolic'
%             optimiser and more than 500 epochs. Only the removed eigenvectors
%             are ever formed; see the notes at local_stream_begin.
%     legacy  everything else. Full per-epoch eigendecomposition, kept, and
%             clean_EEG's cleaning.
%
%   See also GEDAI_per_band, gedai_source, gedai_read, clean_EEG_epoch.

switch action
    case 'begin'
        varargout{1} = local_begin(varargin{:});
    case 'step'
        [varargout{1:max(1, nargout)}] = local_step(varargin{:});
    case 'finish'
        [varargout{1:max(1, nargout)}] = local_finish(varargin{:});
    otherwise
        error('gedai_band_engine:action', 'Unknown action ''%s''.', action);
end
end

% =====================================================================
function state = local_begin(src, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, ...
    optimization_type, parallel, signal_type, minThreshold, maxThreshold, ...
    smoothing_window_seconds, percentile_threshold, rank_truncation, opts) %#ok<INUSL>

N = src.N;
P = src.P;
if P == 0
    error('Cannot process empty data');
end

% Ensure refCOV matches precision of eeg_data
refCOV = cast(refCOV, 'like', zeros(0, 0, src.inputClass));
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;

if isempty(signal_type),  signal_type  = 'eeg'; end
if isempty(minThreshold), minThreshold = 0;     end
if isempty(maxThreshold), maxThreshold = 12;    end
if isempty(smoothing_window_seconds), smoothing_window_seconds = Inf; end
if isempty(rank_truncation), rank_truncation = true; end
if isempty(opts), opts = struct; end
def = struct('want_artifacts', true, 'force_legacy', false, 'artifact_threshold_override', [], ...
             'precision', 'double', 'block_epochs', [], 'parallel_blocks', false, 'verbose', false, ...
             'legacy_artifacts', true);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}), opts.(fn{i}) = def.(fn{i}); end
end
for f = {'block_epochs', 'parallel_blocks', 'verbose'}
    if isempty(opts.(f{1})), opts.(f{1}) = def.(f{1}); end
end

% ---- precision policy ---------------------------------------------------
% 'auto' runs a band in single precision only when its epoch is longer than the
% channel count. Measured on a real night, 256 ch:
%   epoch 384 samples : 1.97x faster, cleaned data differs by 1.3e-5 relative
%   epoch  48 samples : 1.21x faster, cleaned data differs by 3.3e-2 relative
% The second number is not rounding noise. Short epochs go through the Gram
% matrix of the whitened epoch, which squares the condition number, and the
% components that move are the ones sitting near a threshold that SENSAI only
% weakly determines. So the split is deliberate, not a tuning constant.
Te = round(srate * epoch_size);
switch lower(opts.precision)
    case 'single', use_single = true;
    case 'auto',   use_single = Te > N;
    otherwise,     use_single = false;
end
cls = src.inputClass;
cast_back = false;
if use_single && strcmp(cls, 'double')
    cls       = 'single';
    refCOV    = single(refCOV);
    cast_back = true;   % the caller accumulates bands in double
end

% The streaming path pays for itself only when there are many epochs. Its
% threshold stage decomposes up to 500 epochs for SENSAI and then sweeps the
% spectrum of every epoch; when the band has fewer than ~500 epochs those are
% the same epochs twice over, and the legacy path - one decomposition, kept -
% is simply the better algorithm. Long-epoch bands are exactly that case, and
% their stored eigenbasis is small because K = T/Te is small.
epochs_in_band = floor(P / Te);
use_stream = isinf(smoothing_window_seconds) && ~opts.force_legacy && ...
             (ischar(optimization_type) && strcmp(optimization_type, 'parabolic')) && ...
             epochs_in_band > 500;

remainder = rem(P, Te);
pad = 0;
if remainder ~= 0
    pad = Te - remainder;
end

state = struct();
state.N = N;  state.P = P;  state.Te = Te;  state.pad = pad;
state.K = (P + pad) / Te;                 % grid-1 epochs (padded)
state.K2 = state.K - 1;                   % grid-2 epochs
state.sh = Te / 2;                        % grid 2 is offset by half an epoch
state.nEp = floor(P / Te);                % whole epochs inside the signal (ENOVA)
state.cls = cls;
state.cast_back = cast_back;
state.srate = srate;
state.epoch_size = epoch_size;
state.signal_type = signal_type;
state.percentile_threshold = percentile_threshold;
state.want_artifacts = opts.want_artifacts;
state.artifact_threshold_override = opts.artifact_threshold_override;
state.verbose = opts.verbose;
state.done = false;
state.t_clean = 0;

% A wavelet band is reconstructed here, once, in the working precision, and
% read back in slices; reconstructing it per read repeats the synthesis for
% every block the passes touch.
if strcmp(src.kind, 'wavelet')
    src = local_materialize(src, cls);
end

if use_stream
    state.path = 'stream';
    state.src = src;
    state = local_stream_begin(state, refCOV, artifact_threshold_type, optimization_type, ...
        signal_type, minThreshold, maxThreshold, percentile_threshold, rank_truncation, opts);
else
    state.path = 'legacy';
    state.want_artifacts = opts.want_artifacts || opts.legacy_artifacts;
    state = local_legacy_begin(state, src, refCOV, artifact_threshold_type, optimization_type, ...
        parallel, signal_type, minThreshold, maxThreshold, smoothing_window_seconds, ...
        percentile_threshold, rank_truncation);
end
end

% =====================================================================
function src = local_materialize(src, cls)
%LOCAL_MATERIALIZE  The whole band of a wavelet source, as a matrix source.
%   Samples x channels, so that it is built a channel at a time in contiguous
%   memory; blocks are transposed as they are read.
B = gedai_wavelet_band(src.data, src.band, src.level, cls);
inputClass = src.inputClass;
src = gedai_source('matrix_tc', B);
src.inputClass = inputClass;
end

% =====================================================================
function [state, seg, first, last, artifacts] = local_step(state)
if state.done
    error('gedai_band_engine:done', 'The band pass has already delivered all of its output.');
end
tS = tic;
switch state.path
    case 'stream'
        [state, seg, first, last, artifacts] = local_stream_step(state);
    case 'legacy'
        [state, seg, first, last, artifacts] = local_legacy_step(state);
    otherwise
        error('gedai_band_engine:path', 'Unknown path ''%s''.', state.path);
end
state.t_clean = state.t_clean + toc(tS);
end

% =====================================================================
function [SENSAI_score, artifact_threshold_out, ENOVA] = local_finish(state)
if ~state.done
    error('gedai_band_engine:notDone', 'finish called before the last step.');
end
switch state.path
    case 'stream'
        if state.verbose, fprintf('  [stream] cleaning stage: %.2f s\n', state.t_clean); end
        SIGNAL_subspace_similarity = 100 * mean(state.sig_dist);
        NOISE_subspace_similarity  = 100 * mean(state.noi_dist);
        SENSAI_score = SIGNAL_subspace_similarity - (state.noise_multiplier * NOISE_subspace_similarity);
        artifact_threshold_out = state.artifact_threshold;
        if state.nEp > 0
            ENOVA = mean(state.enova_per_epoch);
        else
            ENOVA = 0;
        end
    case 'legacy'
        SENSAI_score = state.SENSAI_score;
        artifact_threshold_out = state.artifact_threshold_out;
        ENOVA = state.ENOVA;
    otherwise
        error('gedai_band_engine:path', 'Unknown path ''%s''.', state.path);
end
end

% =====================================================================
%  STREAM PATH
% =====================================================================
function state = local_stream_begin(state, refCOV, artifact_threshold_type, optimization_type, ...
    signal_type, minThreshold, maxThreshold, percentile_threshold, rank_truncation, opts)
%LOCAL_STREAM_BEGIN  Threshold stage of the streaming implementation.
%
%   The streaming arithmetic rests on two facts about the method:
%
%   1. Only the components ABOVE the threshold are ever used. The cut sits at
%      lambda_98^T1 * exp(5 - t), so in practice a handful of the 47..256
%      components per epoch. Computing and storing the whole eigenbasis for
%      every epoch is ~99% wasted.
%
%   2. The signal covariance never has to be formed. Because the generalized
%      eigenvectors are B-orthonormal, COV = B*V*diag(d)*V'*B over the FULL
%      basis, so the "good" half is just COV minus the "bad" half. SENSAI
%      therefore needs the same handful of vectors the cleaning needs, and
%      COV applied as X*(X'*M)/(T-1).
%
%   The threshold path is deliberately left bit-identical to the full-basis
%   implementation: the SENSAI subsample is drawn with the same stream and the
%   same call, and the global eigenvalue percentile is taken over every epoch
%   via a cheap eigenvalues-only pre-pass rather than estimated from a sample.
%   The SENSAI objective is flat over a wide range of thresholds while the
%   amount of data removed across that same range is not, so an approximate
%   threshold is not a safe trade here.

N = state.N; Te = state.Te; K = state.K; K2 = state.K2; sh = state.sh; cls = state.cls;
src = state.src;

%% ---- regularized reference covariance, factored once -------------------
regularization_lambda = 0.05;
reg_val  = trace(refCOV) / N;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(N, 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;
R_chol = chol(refCOV_reg);

if rank_truncation, r = min(N, Te - 1); else, r = N; end
truncated = (r < N);

%% ---- noise multiplier and SENSAI template (unchanged semantics) --------
if ischar(artifact_threshold_type) && startsWith(artifact_threshold_type, 'auto')
    if strcmp(artifact_threshold_type,'auto+'),     noise_multiplier = 1.5;
    elseif strcmp(artifact_threshold_type,'auto-'), noise_multiplier = 6;
    else,                                           noise_multiplier = 3;
    end
else
    val = artifact_threshold_type;
    if ~isnumeric(val), val = str2double(val); end
    noise_multiplier = 10 - val;
end
if isnan(noise_multiplier), noise_multiplier = 3; end

if strcmpi(signal_type, 'eeg')
    refCOV_top_PCs = 3; SSI_top_PCs = 3;
else
    all_evals_refCOV = sort(eig(refCOV_reg), 'descend');
    cumvar = cumsum(all_evals_refCOV) / sum(all_evals_refCOV);
    refCOV_top_PCs = max(1, min(find(cumvar >= 0.85, 1, 'first'), N - 1));
    SSI_top_PCs = 4;
end
% eigs only accepts double, and refCOV_reg follows the band's working
% precision. The template is a property of the head model, not of the data, so
% it is computed in double regardless and cast afterwards.
[evecs_Template_cov, evals_Template_cov] = eigs(double(refCOV_reg), refCOV_top_PCs);
[~, sidx] = sort(diag(evals_Template_cov), 'descend');
evecs_Template_cov = cast(evecs_Template_cov(:, sidx), 'like', refCOV_reg);

if isempty(percentile_threshold)
    if strcmpi(signal_type, 'eeg'), pct = 98; else, pct = 99; end
else
    pct = percentile_threshold;
end

%% ================= STAGE A : threshold ==================================
tA = tic;

% (A1) SENSAI subsample. Drawn exactly as SENSAI_fminbnd would have drawn it
% from the full set, so the optimiser sees the same epochs in the same order
% and returns the same threshold.
if ~isempty(opts.artifact_threshold_override)
    % Threshold supplied by the caller. Besides making the cleaning stage
    % testable in isolation, this is the only way to give two datasets the
    % same operating point: SENSAI is flat over a wide range, so letting it
    % re-optimise per run makes the amount removed depend on what else is in
    % the file.
    artifact_threshold_scalar = opts.artifact_threshold_override;
else
    max_number_of_epochs = 500;
    if K > max_number_of_epochs
        randStream  = RandStream('mt19937ar', 'Seed', 2);
        sensai_idx  = randperm(randStream, K, max_number_of_epochs);   % order matters
    else
        sensai_idx  = 1:K;
    end
    [Evec_s, Evald_s] = local_gevd_subset(src, cls, sensai_idx, 0, Te, R_chol, r, truncated, N, opts.parallel_blocks);

    switch optimization_type
        case 'parabolic'
            artifact_threshold_scalar = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, ...
                Evald_s, Evec_s, noise_multiplier, evecs_Template_cov, signal_type, ...
                SSI_top_PCs, percentile_threshold);
        otherwise
            error('gedai_band_engine:optimization', ...
                  'Only the ''parabolic'' optimizer is supported on the streaming path.');
    end
    clear Evec_s Evald_s
end

artifact_threshold_scalar = max(minThreshold, min(maxThreshold, artifact_threshold_scalar));
artifact_threshold   = repmat(artifact_threshold_scalar, 1, K);
artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
if isempty(artifact_threshold_2), artifact_threshold_2 = artifact_threshold; end

% (A2) Exact global eigenvalue percentile, per grid, from an
% eigenvalues-only pass. Cheaper than the decomposition with vectors, and it
% reproduces what clean_EEG computed from the full stored spectrum.
% When the epoch is longer than the channel count there is no cheap
% eigenvalues-only route: eig(N x N) without vectors still costs ~60% of the
% full decomposition, so redoing it in the cleaning stage would be a net loss.
% In that regime the pre-pass keeps the leading eigenvectors as well, and the
% cleaning stage becomes pure application. Those bands have few epochs
% (K = T/Te), so the cache is small. For the short-epoch bands the opposite
% holds: the Gram route is cheap and K is large, so nothing is kept.
keep_vectors = ~truncated;
k_keep = min(r, 32);

[Evald_all_1, Utop_1, Dtop_1] = local_gevd_prepass(src, cls, 0, Te, R_chol, r, truncated, N, K, opts, keep_vectors, k_keep);
T1_1 = (105 - artifact_threshold) / 100;
cut_1 = gedai_eig_threshold(Evald_all_1, N, pct, T1_1);
clear Evald_all_1

if K2 > 0
    [Evald_all_2, Utop_2, Dtop_2] = local_gevd_prepass(src, cls, sh, Te, R_chol, r, truncated, N, K2, opts, keep_vectors, k_keep);
    T1_2 = (105 - artifact_threshold_2) / 100;
    cut_2 = gedai_eig_threshold(Evald_all_2, N, pct, T1_2);
    clear Evald_all_2
else
    cut_2 = []; Utop_2 = []; Dtop_2 = [];
end
if opts.verbose, fprintf('  [stream] threshold stage: %.2f s\n', toc(tA)); end

%% ---- cleaning-stage set-up ---------------------------------------------
state.R_chol = R_chol;
state.B = refCOV_reg;
state.r = r;
state.truncated = truncated;
state.cut_1 = cut_1;  state.Utop_1 = Utop_1;  state.Dtop_1 = Dtop_1;
state.cut_2 = cut_2;  state.Utop_2 = Utop_2;  state.Dtop_2 = Dtop_2;
state.cosine_weights = create_cosine_weights(N, state.srate, state.epoch_size, 1);
state.noise_multiplier = noise_multiplier;
state.artifact_threshold = artifact_threshold;

% SENSAI accumulators (grid 1 only)
state.M_ssi = min(size(evecs_Template_cov, 2), SSI_top_PCs);
state.Template_guess = evecs_Template_cov(:, 1:state.M_ssi);
state.T_proj = refCOV_reg * state.Template_guess;
state.sig_dist = zeros(1, K);
state.noi_dist = zeros(1, K);

% Blocks: keep the per-block working set near 256 MB regardless of band
block = opts.block_epochs;
if isempty(block)
    bytes = 8; if strcmp(cls, 'single'), bytes = 4; end
    block = max(1, min(512, floor(256*2^20 / max(6 * N * Te * bytes, 1))));
end
state.block = block;
state.nblocks_1 = ceil(K / block);
state.nblocks_2 = ceil(K2 / block);
% A grid goes to the pool only when it has more than one block, as before: a
% single block runs on the client, whose multithreaded BLAS is not bit-equal
% to a single-threaded worker's.
state.par_1 = opts.parallel_blocks && state.nblocks_1 > 1;
state.par_2 = opts.parallel_blocks && state.nblocks_2 > 1;
if state.par_1 || state.par_2
    p = gcp('nocreate');
    if isempty(p), nw = 0; else, nw = p.NumWorkers; end
    % One wave dispatches nw blocks of each grid, 2*nw tasks - the same task
    % count per parfor as the old grid-at-a-time waves of 2*nw blocks.
    state.wave = max(1, nw);
else
    state.wave = 1;
end
state.next_block = 1;
state.carry = [];                          % grid-2 samples past the last output

% ENOVA per epoch, in the chunking the matrix code used: var() of a single
% column is not bit-equal to the same column inside a wider call.
state.enova_per_epoch = zeros(1, state.nEp);
state.ep_per_chunk = max(1, floor(64*2^20 / max(N * Te * 8, 1)));
state.enova_next = 1;
state.pend_O = zeros(N, 0, cls);          % signal of epochs enova_next..
state.pend_C = zeros(N, 0, cls);          % cleaned output of the same epochs
end

% =====================================================================
function [state, seg, first, last, artifacts] = local_stream_step(state)
%LOCAL_STREAM_STEP  Clean the next wave of blocks of both grids; emit their output.
N = state.N; Te = state.Te; K = state.K; K2 = state.K2; sh = state.sh; cls = state.cls;
block = state.block;

b0 = state.next_block;
b1 = min(state.nblocks_1, b0 + state.wave - 1);
blocks_1 = b0:b1;
if K2 > 0 && b0 <= state.nblocks_2
    blocks_2 = b0:min(state.nblocks_2, b1);
else
    blocks_2 = zeros(1, 0);
end
E0 = (b0 - 1) * block + 1;  E1 = min(K, b1 * block);

% ---- task list: one task per block per grid ----------------------------
grid = [ones(1, numel(blocks_1)), 2 * ones(1, numel(blocks_2))];
blk  = [blocks_1, blocks_2];
nT   = numel(blk);
css  = zeros(1, nT); ces = zeros(1, nT);
Xb   = cell(1, nT); Ub = cell(1, nT); Db = cell(1, nT);
for j = 1:nT
    if grid(j) == 1
        Kg = K; off = 0; Utop = state.Utop_1; Dtop = state.Dtop_1;
    else
        Kg = K2; off = sh; Utop = state.Utop_2; Dtop = state.Dtop_2;
    end
    css(j) = (blk(j) - 1) * block + 1;
    ces(j) = min(Kg, blk(j) * block);
    s0 = off + (css(j) - 1) * Te;
    Xb{j} = gedai_read(state.src, s0 + 1, s0 + (ces(j) - css(j) + 1) * Te, cls);
    if ~isempty(Utop)
        Ub{j} = Utop(:, :, css(j):ces(j));
        Db{j} = Dtop(:, css(j):ces(j));
    end
end
clear Utop Dtop

onPool = (grid == 1 & state.par_1) | (grid == 2 & state.par_2);
segs = cell(1, nT); sgs = cell(1, nT); ngs = cell(1, nT);

% Everything the parfor body touches is a local here, so that no part of the
% state struct (the band and the eigenvector caches in particular) is broadcast.
R_chol = state.R_chol; B = state.B; r = state.r; truncated = state.truncated;
cut_1 = state.cut_1; cut_2 = state.cut_2; cw = state.cosine_weights;
Template_guess = state.Template_guess; T_proj = state.T_proj; M_ssi = state.M_ssi;

pj = find(onPool);
if ~isempty(pj)
    pX = Xb(pj); pU = Ub(pj); pD = Db(pj); pcs = css(pj); pce = ces(pj); pg = grid(pj);
    m = numel(pj);
    pseg = cell(1, m); psg = cell(1, m); png = cell(1, m);
    parfor j = 1:m
        if pg(j) == 1
            [pseg{j}, psg{j}, png{j}] = local_clean_block(pX{j}, pcs(j), pce(j), K, Te, R_chol, B, r, ...
                truncated, N, cut_1, cw, true, Template_guess, T_proj, M_ssi, pU{j}, pD{j});
        else
            [pseg{j}, psg{j}, png{j}] = local_clean_block(pX{j}, pcs(j), pce(j), K2, Te, R_chol, B, r, ...
                truncated, N, cut_2, cw, false, [], [], M_ssi, pU{j}, pD{j});
        end
    end
    segs(pj) = pseg; sgs(pj) = psg; ngs(pj) = png;
    clear pX pU pD pseg
end
for j = find(~onPool)
    if grid(j) == 1
        [segs{j}, sgs{j}, ngs{j}] = local_clean_block(Xb{j}, css(j), ces(j), K, Te, R_chol, B, r, ...
            truncated, N, cut_1, cw, true, Template_guess, T_proj, M_ssi, Ub{j}, Db{j});
    else
        [segs{j}, sgs{j}, ngs{j}] = local_clean_block(Xb{j}, css(j), ces(j), K2, Te, R_chol, B, r, ...
            truncated, N, cut_2, cw, false, [], [], M_ssi, Ub{j}, Db{j});
    end
end
clear Ub Db

% ---- SENSAI accumulators ------------------------------------------------
for j = find(grid == 1)
    state.sig_dist(css(j):ces(j)) = sgs{j};
    state.noi_dist(css(j):ces(j)) = ngs{j};
end

% ---- grid 1, plus grid 2 wherever it overlaps --------------------------
% The matrix code built both grids over the whole band and added them once,
% cleaned_1(:, sh+1:sh+n2) + cleaned_2. Per sample that is c1 + c2 where grid 2
% covers the sample and c1 alone at the two ends of the band. Grid-2 epoch e
% spans grid-1 samples (e-1)*Te+sh+1 .. e*Te+sh, so the second half of the last
% grid-2 epoch of this wave belongs to the next wave and is carried over.
is1 = grid == 1;
out  = [segs{is1}];                      % grid 1, epochs E0..E1
base = (E0 - 1) * Te;
if E0 > 1
    % carried second half of grid-2 epoch E0-1
    out(:, 1:sh) = out(:, 1:sh) + state.carry;
end
state.carry = [];
if ~isempty(blocks_2)
    F0 = css(find(~is1, 1)); F1 = ces(find(~is1, 1, 'last'));
    S2 = [segs{~is1}];
    % Edge weighting of the shifted grid
    if F0 == 1
        S2(:, 1:sh) = S2(:, 1:sh) .* cw(:, 1:sh);
    end
    if F1 == K2
        e2 = size(S2, 2) - sh;
        S2(:, e2+1:end) = S2(:, e2+1:end) .* cw(:, sh+1:end);
    end
    s2first  = (F0 - 1) * Te + sh + 1;                 % global sample of S2(:, 1)
    inRegion = min(E1 * Te, F1 * Te + sh) - s2first + 1;
    ix = (s2first - base) : (s2first - base + inRegion - 1);
    out(:, ix) = out(:, ix) + S2(:, 1:inRegion);
    if inRegion < size(S2, 2)
        state.carry = S2(:, inRegion+1:end);
    end
    clear S2
end
clear segs

% ---- ENOVA per epoch, accumulated without ever building the artifact array ----
% cleaned + artifacts == input by construction, so the removed signal is just
% the difference and never needs to be stored in full.
nEp = state.nEp;
if E0 <= nEp
    nNew = (min(E1, nEp) - E0 + 1) * Te;
    state.pend_O = [state.pend_O, Xb{is1}];          % grid-1 signal of this wave
    if nNew < size(out, 2)
        state.pend_O = state.pend_O(:, 1:end - (size(out, 2) - nNew));
        state.pend_C = [state.pend_C, out(:, 1:nNew)];
    else
        state.pend_C = [state.pend_C, out];
    end
    epc = state.ep_per_chunk;
    used = 0;
    while state.enova_next <= nEp
        cs = state.enova_next;
        ce = min(cs + epc - 1, nEp);
        if ce > min(E1, nEp), break; end
        nc = ce - cs + 1;
        sl = used + 1 : used + nc * Te;
        O  = reshape(state.pend_O(:, sl), [], nc);
        A  = O - reshape(state.pend_C(:, sl), [], nc);
        vo = var(O, 0, 1); va = var(A, 0, 1);
        e  = zeros(1, nc); msk = vo > 0; e(msk) = va(msk) ./ vo(msk);
        state.enova_per_epoch(cs:ce) = e;
        state.enova_next = ce + 1;
        used = used + nc * Te;
    end
    if used > 0
        state.pend_O = state.pend_O(:, used+1:end);
        state.pend_C = state.pend_C(:, used+1:end);
    end
end

% ---- emit, clipped to the unpadded signal -----------------------------
first = base + 1;
last  = min(state.P, E1 * Te);
if last < E1 * Te
    out = out(:, 1:last - first + 1);
end
seg = out;
if state.want_artifacts
    artifacts = gedai_read(state.src, first, last, cls) - seg;
else
    artifacts = [];
end

state.next_block = b1 + 1;
state.done = b1 >= state.nblocks_1;
if state.done
    state.pend_O = []; state.pend_C = []; state.src = [];
    state.Utop_1 = []; state.Dtop_1 = []; state.Utop_2 = []; state.Dtop_2 = [];
end
end

% =====================================================================
function [seg, sig_b, noi_b, nb_sum, nb_max] = local_clean_block( ...
    Xblk, cs, ce, K, Te, R_chol, B, r, truncated, N, cut, cosine_weights, ...
    do_sensai, Template_guess, T_proj, M_ssi, Utop, Dtop)
%LOCAL_CLEAN_BLOCK  One block of epochs, given that block's samples.
%   Depends on nothing outside itself except the (already fixed) threshold and
%   the two global epoch indices used by the cosine edge rule, which is what
%   makes the block loop safe to run on a pool.

c  = ce - cs + 1;
D3 = reshape(Xblk, N, Te, c);
Dc = D3 - mean(D3, 2);
have_cache = ~isempty(Utop);
if have_cache
    Z = [];      % vectors already available; only Dc is needed below
else
    Z = reshape(R_chol' \ reshape(Dc, N, []), N, Te, c);   % one batched triangular solve
    if ~truncated
        A = pagemtimes(Z, 'none', Z, 'transpose') / (Te - 1);
    end
end

seg    = zeros(N, c*Te, 'like', Xblk);
sig_b  = zeros(1, c);
noi_b  = zeros(1, c);
nb_sum = 0; nb_max = 0;
half   = Te / 2;
tol    = 1e-12;

for k = 1:c
    g = cs + k - 1;                       % global epoch index in this grid
    if have_cache
        dk  = Dtop(:, k);                 % already sorted descending
        bad = find(abs(dk) >= cut(g));
        if numel(bad) == numel(dk)
            % cache exhausted: the cut fell below the k-th eigenvalue, so
            % redo this epoch in full rather than silently under-removing
            Zk = R_chol' \ Dc(:,:,k);
            Ak = Zk * Zk' / (Te - 1); Ak = (Ak + Ak') / 2;
            [Wa, Da] = eig(Ak);
            d = diag(Da); [d, ix] = sort(d, 'descend');
            bad = find(abs(d) >= cut(g));
            U = Wa(:, ix(bad));
            d_all = d;
        else
            U = Utop(:, bad, k);
            d_all = dk;
        end
    elseif truncated
        Zk = Z(:,:,k);
        G  = Zk' * Zk; G = (G + G') / 2;
        [Wg, Dg] = eig(G);
        dg = diag(Dg);
        [dg, ix] = sort(dg, 'descend');
        d = dg(1:r) / (Te - 1);
        bad = find(abs(d) >= cut(g));
        if ~isempty(bad)
            U = (Zk * Wg(:, ix(bad))) ./ sqrt(max(dg(bad), realmin)).';
        end
        d_all = d;
    else
        Ak = A(:,:,k); Ak = (Ak + Ak') / 2;
        [Wa, Da] = eig(Ak);
        d = diag(Da);
        [d, ix] = sort(d, 'descend');
        bad = find(abs(d) >= cut(g));
        if ~isempty(bad), U = Wa(:, ix(bad)); end
        d_all = d;
    end

    num_bad = numel(bad);
    nb_sum  = nb_sum + num_bad;
    nb_max  = max(nb_max, num_bad);

    Xk = D3(:,:,k);
    if num_bad > 0
        V_bad = R_chol \ U;               % back-transform only what is removed
        d_bad = abs(d_all(bad));
        if num_bad <= Te
            Xk = Xk - (B * V_bad) * (V_bad' * Xk);
        else
            Xk = Xk - B * (V_bad * (V_bad' * Xk));
        end
    else
        V_bad = zeros(N, 0, 'like', Xblk); d_bad = zeros(0, 1, 'like', Xblk);
    end

    if do_sensai
        [sig_b(k), noi_b(k)] = local_sensai_epoch(Dc(:,:,k), Te, V_bad, d_bad, ...
            B, T_proj, Template_guess, M_ssi, N, tol);
    end

    % cosine edge rule, identical to clean_EEG
    if g == 1
        Xk(:, half+1:end) = Xk(:, half+1:end) .* cosine_weights(:, half+1:end);
    elseif g == K
        Xk(:, 1:half)     = Xk(:, 1:half)     .* cosine_weights(:, 1:half);
    else
        Xk = Xk .* cosine_weights;
    end
    seg(:, (k-1)*Te+1 : k*Te) = Xk;
end
end

% =====================================================================
function [sig, noi] = local_sensai_epoch(Xc, Te, V_bad, d_bad, B, T_proj, ...
                                         Template_guess, M, N, tol)
%LOCAL_SENSAI_EPOCH  One epoch's SENSAI subspace similarities, from the bad
%   eigenpairs only.
%
%   The full-basis code applied  cov_signal * Y = B*V_good*diag(d_good)*V_good'*Y.
%   Summed over ALL components that operator equals COV*B^{-1}, and both call
%   sites pass Y = B*something, so
%       signal_op(B*S) = COV*S - B*(V_bad*(d_bad.*(V_bad'*(B*S)))).
%   COV*S is applied as Xc*(Xc'*S)/(Te-1) and never formed.

num_bad = numel(d_bad);

% --- signal subspace ---
S1 = Template_guess;
Y1 = local_cov_apply(Xc, Te, S1) - B * (V_bad * (d_bad .* (V_bad' * (B * S1))));
[Q1, ~] = qr(Y1, 0);
Y2 = local_cov_apply(Xc, Te, Q1) - B * (V_bad * (d_bad .* (V_bad' * (B * Q1))));
[evecs_signal, ~] = qr(Y2, 0);
sig = abs(det(evecs_signal' * Template_guess));

% --- noise subspace ---
if num_bad >= M
    Y1n = B * (V_bad * (d_bad .* (V_bad' * T_proj)));
    [Q1n, ~] = qr(Y1n, 0);
    Tn  = B * Q1n;
    Y2n = B * (V_bad * (d_bad .* (V_bad' * Tn)));
    [evecs_noise, ~] = qr(Y2n, 0);
    noi = abs(det(evecs_noise' * Template_guess));
elseif num_bad > 0
    V_rows = V_bad' * B;
    cov_noise = V_rows' * (V_rows .* d_bad);
    cov_noise = (cov_noise + cov_noise') / 2;
    if max(abs(cov_noise(:))) < tol
        Y1n = eye(N, M, 'like', Xc);
    else
        Y1n = cov_noise * Template_guess;
    end
    [Q1n, ~] = qr(Y1n, 0);
    Y2n = cov_noise * Q1n;
    [evecs_noise, ~] = qr(Y2n, 0);
    noi = abs(det(evecs_noise' * Template_guess));
else
    evecs_noise = eye(N, M, 'like', Xc);
    noi = abs(det(evecs_noise' * Template_guess));
end
end

function Y = local_cov_apply(Xc, Te, S)
Y = (Xc * (Xc' * S)) / (Te - 1);
end

% =====================================================================
function [Evec, Evald] = local_gevd_subset(src, cls, idx, off, Te, R_chol, r, truncated, N, parallel_blocks)
%LOCAL_GEVD_SUBSET  Full decomposition, for the SENSAI epochs only. The
%   optimiser sweeps thresholds, so the good/bad split moves and it genuinely
%   needs the whole eigenbasis on these few hundred epochs. Several seconds a
%   band, so it goes to the pool as well when one is in use.
n = numel(idx);
Xs = zeros(N, Te, n, cls);
for j = 1:n
    s0 = off + (idx(j)-1)*Te;
    Xs(:,:,j) = gedai_read(src, s0+1, s0+Te, cls);
end

if parallel_blocks && n > 8
    p = gcp('nocreate');
    if isempty(p), nw = 1; else, nw = p.NumWorkers; end
    part = max(1, ceil(n / max(nw, 1)));
    starts = 1:part:n;
    m = numel(starts);
    Xc = cell(1, m); lo = zeros(1, m); hi = zeros(1, m);
    for j = 1:m
        lo(j) = starts(j); hi(j) = min(n, starts(j)+part-1);
        Xc{j} = Xs(:,:,lo(j):hi(j));
    end
    clear Xs
    Ec = cell(1, m); Dc = cell(1, m);
    parfor j = 1:m
        [Ec{j}, Dc{j}] = local_gevd_subset_core(Xc{j}, Te, R_chol, r, truncated, N);
    end
    Evec  = zeros(N, r, n, cls);
    Evald = zeros(r, n, cls);
    for j = 1:m
        Evec(:,:,lo(j):hi(j)) = Ec{j};
        Evald(:,lo(j):hi(j))  = Dc{j};
    end
else
    [Evec, Evald] = local_gevd_subset_core(Xs, Te, R_chol, r, truncated, N);
end
end

function [Evec, Evald] = local_gevd_subset_core(Xs, Te, R_chol, r, truncated, N)
n = size(Xs, 3);
Evec  = zeros(N, r, n, 'like', Xs);
Evald = zeros(r, n, 'like', Xs);
for j = 1:n
    Xk = Xs(:,:,j);
    Xk = Xk - mean(Xk, 2);
    Z  = R_chol' \ Xk;
    if truncated
        [U, S, ~] = svd(Z, 'econ');
        s = diag(S);
        Evec(:,:,j) = R_chol \ U(:, 1:r);
        Evald(:,j)  = s(1:r).^2 / (Te - 1);
    else
        A = Z * Z' / (Te - 1); A = (A + A') / 2;
        [U, D] = eig(A);
        Evec(:,:,j) = R_chol \ U;
        Evald(:,j)  = diag(D);
    end
end
end

% =====================================================================
function [Evald, Utop, Dtop] = local_gevd_prepass(src, cls, off, Te, R_chol, r, truncated, N, K, opts, keep_vectors, k_keep)
%LOCAL_GEVD_PREPASS  Spectrum of every epoch, so that the global percentile is
%   exact rather than sampled. Optionally also keeps the leading k_keep
%   whitened eigenvectors, for the regime where recomputing them in the
%   cleaning stage would cost more than storing them.
Evald = zeros(r, K, cls);
if keep_vectors
    Utop = zeros(N, k_keep, K, cls);
    Dtop = zeros(k_keep, K, cls);
else
    Utop = []; Dtop = [];
end
block = opts.block_epochs;
if isempty(block)
    bytes = 8; if strcmp(cls, 'single'), bytes = 4; end
    block = max(1, min(512, floor(256*2^20 / max(4 * N * Te * bytes, 1))));
end
starts  = 1:block:K;
nblocks = numel(starts);

if opts.parallel_blocks && nblocks > 1
    p = gcp('nocreate');
    if isempty(p), nw = 0; else, nw = p.NumWorkers; end
    wave = max(1, 2 * max(nw, 1));
    for w0 = 1:wave:nblocks
        w1 = min(nblocks, w0 + wave - 1);
        m  = w1 - w0 + 1;
        Xb = cell(1, m); css = zeros(1, m); ces = zeros(1, m);
        for j = 1:m
            css(j) = starts(w0 + j - 1);
            ces(j) = min(css(j) + block - 1, K);
            s0 = off + (css(j)-1)*Te;
            Xb{j} = gedai_read(src, s0+1, s0 + (ces(j)-css(j)+1)*Te, cls);
        end
        Eb = cell(1, m); Uc = cell(1, m); Dc_ = cell(1, m);
        parfor j = 1:m
            [Eb{j}, Uc{j}, Dc_{j}] = local_prepass_block(Xb{j}, Te, R_chol, r, ...
                truncated, N, keep_vectors, k_keep);
        end
        for j = 1:m
            Evald(:, css(j):ces(j)) = Eb{j};
            if keep_vectors
                Utop(:, :, css(j):ces(j)) = Uc{j};
                Dtop(:, css(j):ces(j))    = Dc_{j};
            end
        end
    end
else
    for b = 1:nblocks
        cs = starts(b);
        ce = min(cs + block - 1, K);
        s0 = off + (cs-1)*Te;
        Xblk = gedai_read(src, s0+1, s0 + (ce-cs+1)*Te, cls);
        [Eb, Uc, Dc_] = local_prepass_block(Xblk, Te, R_chol, r, truncated, N, keep_vectors, k_keep);
        Evald(:, cs:ce) = Eb;
        if keep_vectors
            Utop(:, :, cs:ce) = Uc;
            Dtop(:, cs:ce)    = Dc_;
        end
    end
end
end

% =====================================================================
function [Evald, Utop, Dtop] = local_prepass_block(Xblk, Te, R_chol, r, truncated, N, keep_vectors, k_keep)
%LOCAL_PREPASS_BLOCK  Spectrum (and optionally leading vectors) for one block.
c  = size(Xblk, 2) / Te;
D3 = reshape(Xblk, N, Te, c);
Dc = D3 - mean(D3, 2);
Z  = reshape(R_chol' \ reshape(Dc, N, []), N, Te, c);
Evald = zeros(r, c, 'like', Xblk);
if keep_vectors
    Utop = zeros(N, k_keep, c, 'like', Xblk);
    Dtop = zeros(k_keep, c, 'like', Xblk);
else
    Utop = []; Dtop = [];
end
if truncated
    for k = 1:c
        Zk = Z(:,:,k); G = Zk' * Zk; G = (G + G') / 2;
        dg = sort(eig(G), 'descend');
        Evald(:, k) = dg(1:r) / (Te - 1);
    end
else
    A = pagemtimes(Z, 'none', Z, 'transpose') / (Te - 1);
    for k = 1:c
        Ak = A(:,:,k); Ak = (Ak + Ak') / 2;
        if keep_vectors
            [Wa, Da] = eig(Ak);
            d = diag(Da); [d, ix] = sort(d, 'descend');
            Evald(:, k)  = d;
            Utop(:, :, k) = Wa(:, ix(1:k_keep));
            Dtop(:, k)    = d(1:k_keep);
        else
            Evald(:, k) = sort(eig(Ak), 'descend');
        end
    end
end
end

% =====================================================================
%  LEGACY PATH
% =====================================================================
function state = local_legacy_begin(state, src, refCOV, artifact_threshold_type, optimization_type, ...
    parallel, signal_type, minThreshold, maxThreshold, smoothing_window_seconds, ...
    percentile_threshold, rank_truncation)
%LOCAL_LEGACY_BEGIN  The full-basis implementation, run to completion.
%   Decomposition, threshold, cleaning of both grids with their overlap-add,
%   ENOVA and the final SENSAI score all happen here; the steps only hand out
%   slices of the cleaned band. The band's input, cleaned output and artifacts
%   are the only band-sized arrays, and the input is released before ENOVA.

N_EEG_electrodes = state.N; epoch_samples = state.Te; cls = state.cls;
N_epochs = state.K;

if state.K2 == 0
    % The matrix code failed here too, one step later: the shifted grid is
    % empty and its edge weighting indexed past the end of it.
    error('GEDAI_per_band:singleEpoch', ...
        'Band has a single epoch of %d samples; at least two are needed.', epoch_samples);
end

%% Generalized Eigendecomposition (GEVD)
regularization_lambda = 0.05;
reg_val = trace(refCOV) / N_EEG_electrodes;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(N_EEG_electrodes, 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;
R_chol = chol(refCOV_reg);

if rank_truncation
    gevd_rank = min(N_EEG_electrodes, epoch_samples - 1);
else
    gevd_rank = N_EEG_electrodes;
end

[Evec,   Evald]   = local_gevd(src, cls, 0,        N_epochs,   R_chol, gevd_rank, N_EEG_electrodes, epoch_samples);
[Evec_2, Evald_2] = local_gevd(src, cls, state.sh, state.K2,  R_chol, gevd_rank, N_EEG_electrodes, epoch_samples);

%% Determine Noise Multiplier and Optimization Parameters
if ischar(artifact_threshold_type) && startsWith(artifact_threshold_type, 'auto')
    if strcmp(artifact_threshold_type,'auto+'), noise_multiplier = 1.5;
    elseif strcmp(artifact_threshold_type,'auto'), noise_multiplier = 3;
    elseif strcmp(artifact_threshold_type,'auto-'), noise_multiplier = 6;
    else, noise_multiplier = 3;
    end
else
    if isnumeric(artifact_threshold_type)
        val = artifact_threshold_type;
    else
        val = str2double(artifact_threshold_type);
    end
    noise_multiplier = 10 - val;
end
if isnan(noise_multiplier), noise_multiplier = 3; end

if strcmpi(signal_type, 'eeg')
   refCOV_top_PCs = 3;
   SSI_top_PCs = 3;
elseif strcmpi(signal_type, 'meg')
        all_evals_refCOV = eig(refCOV_reg);
        all_evals_refCOV = sort(all_evals_refCOV, 'descend');
        cumvar_refCOV = cumsum(all_evals_refCOV) / sum(all_evals_refCOV);
        refCOV_top_PCs = find(cumvar_refCOV >= 0.85, 1, 'first');
        refCOV_top_PCs = max(1, min(refCOV_top_PCs, N_EEG_electrodes - 1));
        SSI_top_PCs = 4;
end

if refCOV_top_PCs < SSI_top_PCs
    warning('GEDAI:LowRefCOVPCs', 'refCOV variance appears to be concentrated in too few principal components. Verify that leadfield matrix is well constructed.');
end

% eigs only accepts double; refCOV_reg follows the band's working precision.
[evecs_Template_cov, evals_Template_cov] = eigs(double(refCOV_reg), refCOV_top_PCs);
[~, sidxS_Template_cov] = sort(diag(evals_Template_cov), 'descend');
evecs_Template_cov = cast(evecs_Template_cov(:, sidxS_Template_cov), 'like', refCOV_reg);

epoch_size = state.epoch_size;
if isinf(smoothing_window_seconds)
    window_seconds = N_epochs * epoch_size;
else
    window_seconds = smoothing_window_seconds;
end
window_epochs = max(1, round(window_seconds / epoch_size));
step_epochs = max(1, round(window_epochs / 2));

num_windows = max(1, ceil((N_epochs - window_epochs) / step_epochs) + 1);
if N_epochs <= window_epochs
    num_windows = 1;
    window_epochs = N_epochs;
end

window_centers = zeros(1, num_windows);
optimal_threshold_per_window = zeros(1, num_windows);

if ~isempty(state.artifact_threshold_override)
    optimal_threshold_per_window(:) = state.artifact_threshold_override;
    window_centers = ((0:num_windows-1) * step_epochs) + 1;
    num_windows_loop = 0;
else
    num_windows_loop = num_windows;
end

for w = 1:num_windows_loop
    idx_start = (w - 1) * step_epochs + 1;
    idx_end = min(N_epochs, idx_start + window_epochs - 1);
    if w == num_windows && (idx_end - idx_start + 1) < window_epochs/2 && num_windows > 1
        idx_start = max(1, N_epochs - window_epochs + 1);
        idx_end = N_epochs;
    end
    window_centers(w) = (idx_start + idx_end) / 2;

    Evald_sub = Evald(:,idx_start:idx_end);
    Evec_sub = Evec(:,:,idx_start:idx_end);

    switch optimization_type
        case 'parabolic'
            [optimal_artifact_threshold] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Evald_sub, Evec_sub, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold);

        case 'grid'
            automatic_thresholding_step_size = 1/3;
            AutomaticThresholdSweep = minThreshold:automatic_thresholding_step_size:maxThreshold;
            SIGNAL_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
            NOISE_subspace_similarity = zeros(1, length(AutomaticThresholdSweep));
            SENSAI_score = zeros(1, length(AutomaticThresholdSweep));
            if parallel
                parfor threshold_index=1:length(AutomaticThresholdSweep)
                    artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                    [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Evald_sub, Evec_sub, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold);
                end
            else
                for threshold_index=1:length(AutomaticThresholdSweep)
                    artifact_threshold_iter = AutomaticThresholdSweep(threshold_index);
                    [SIGNAL_subspace_similarity(threshold_index), NOISE_subspace_similarity(threshold_index), SENSAI_score(threshold_index)] = SENSAI(artifact_threshold_iter, refCOV, Evald_sub, Evec_sub, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold);
                end
            end
            [~, SENSAI_index] = max(SENSAI_score);
            NOISE_changepoint_index = findchangepts(diff(smoothdata(NOISE_subspace_similarity, "movmean",6)),Statistic="mean", MaxNumChanges=2);
            if isempty(NOISE_changepoint_index)
                NOISE_changepoint_index = length(AutomaticThresholdSweep);
            end
            if SENSAI_index > NOISE_changepoint_index(1)
                optimal_artifact_threshold = AutomaticThresholdSweep(NOISE_changepoint_index(1));
            else
                optimal_artifact_threshold = AutomaticThresholdSweep(SENSAI_index);
            end
        otherwise
            error('gedai_band_engine:optimization', 'Unknown optimization type ''%s''.', optimization_type);
    end
    optimal_threshold_per_window(w) = optimal_artifact_threshold;
end

if num_windows > 1
    if num_windows >= 3
        optimal_threshold_per_window = smoothdata(optimal_threshold_per_window, 'movmean', 3);
    end
    padded_centers    = [1, window_centers, N_epochs];
    padded_thresholds = [optimal_threshold_per_window(1), optimal_threshold_per_window, optimal_threshold_per_window(end)];
    [unique_centers, unique_idx] = unique(padded_centers);
    unique_thresholds = padded_thresholds(unique_idx);
    artifact_threshold_array = interp1(unique_centers, unique_thresholds, 1:N_epochs, 'makima');
else
    artifact_threshold_array = repmat(optimal_threshold_per_window, 1, N_epochs);
end

artifact_threshold_array = max(minThreshold, min(maxThreshold, artifact_threshold_array));
artifact_threshold = artifact_threshold_array;
cosine_weights = create_cosine_weights(N_EEG_electrodes, state.srate, epoch_size, 1);

artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
if isempty(artifact_threshold_2)
    artifact_threshold_2 = artifact_threshold;
end

%% Cleaning: clean_EEG per epoch, straight into the overlap-added band
% The matrix code built grid 1 and grid 2 in full, edge-weighted grid 2, added
% it at an offset of half an epoch and truncated to the signal. Here each
% grid-2 epoch (edge-weighted first, as before) is added where it lands and
% samples beyond the signal are never stored; every output sample is still
% c1 + c2 (or c1 alone), computed once.
P = state.P; sh = state.sh; K2 = state.K2;
len = state.nEp * epoch_samples;     % whole epochs inside the signal, for ENOVA
keep_tail = state.want_artifacts;
C  = zeros(N_EEG_electrodes, P, cls);
A  = zeros(N_EEG_electrodes, len, cls);                 % artifacts of the whole epochs
At = zeros(N_EEG_electrodes, (P - len) * keep_tail, cls); % the rest, only for the caller

[thr_1, artifact_threshold_out, ~, mag_1] = clean_EEG_thresholds( ...
    Evald, N_EEG_electrodes, artifact_threshold, refCOV, signal_type, refCOV_reg, percentile_threshold);
for i = 1:N_epochs
    s0 = (i-1) * epoch_samples;
    X  = gedai_read(src, s0 + 1, s0 + epoch_samples, cls);
    [cl, ar] = clean_EEG_epoch(X, i, N_epochs, mag_1, thr_1, Evec, refCOV_reg, cosine_weights, epoch_samples);
    n  = min(epoch_samples, P - s0);
    nA = max(0, min(n, len - s0));
    C(:, s0+1:s0+n)  = cl(:, 1:n);
    A(:, s0+1:s0+nA) = ar(:, 1:nA);
    if keep_tail && nA < n
        At(:, s0+nA+1-len : s0+n-len) = ar(:, nA+1:n);
    end
end

[thr_2, ~, ~, mag_2] = clean_EEG_thresholds( ...
    Evald_2, N_EEG_electrodes, artifact_threshold_2, refCOV, signal_type, refCOV_reg, percentile_threshold);
clear Evald_2
for i = 1:K2
    s0 = sh + (i-1) * epoch_samples;
    X  = gedai_read(src, s0 + 1, s0 + epoch_samples, cls);
    [cl, ar] = clean_EEG_epoch(X, i, K2, mag_2, thr_2, Evec_2, refCOV_reg, cosine_weights, epoch_samples);
    % Edge weighting of the shifted grid, cleaned then artifacts as before
    if i == 1
        cl(:, 1:sh) = cl(:, 1:sh) .* cosine_weights(:, 1:sh);
    end
    if i == K2
        cl(:, sh+1:end) = cl(:, sh+1:end) .* cosine_weights(:, (sh+1):end);
    end
    if i == 1
        ar(:, 1:sh) = ar(:, 1:sh) .* cosine_weights(:, 1:sh);
    end
    if i == K2
        ar(:, sh+1:end) = ar(:, sh+1:end) .* cosine_weights(:, (sh+1):end);
    end
    n  = min(epoch_samples, P - s0);
    nA = max(0, min(n, len - s0));
    C(:, s0+1:s0+n)  = C(:, s0+1:s0+n)  + cl(:, 1:n);
    A(:, s0+1:s0+nA) = A(:, s0+1:s0+nA) + ar(:, 1:nA);
    if keep_tail && nA < n
        t = s0+nA+1-len : s0+n-len;
        At(:, t) = At(:, t) + ar(:, nA+1:n);
    end
end
clear Evec_2 mag_2 thr_2 X cl ar src

%% Calculate final SENSAI score
[~, ~, SENSAI_score] = SENSAI(mean(artifact_threshold_out), refCOV, Evald, Evec, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold);
clear Evec Evald

%% ENOVA: one var() over all whole epochs of original = cleaned + artifacts, and of artifacts
% var() allocates one temporary of its input's size. Unless the caller wants
% the artifacts, their array is turned into the original signal in place once
% their own var() is done.
num_epochs = state.nEp;
var_art = var(reshape(A, [], num_epochs), 0, 1);
if state.want_artifacts
    var_orig = var(reshape(C(:, 1:len) + A, [], num_epochs), 0, 1);
    A = [A, At];
else
    samples_per_piece = max(1, floor(256 * 2^20 / (8 * N_EEG_electrodes)));
    for s0 = 0:samples_per_piece:len-1
        cols = s0+1 : min(len, s0 + samples_per_piece);
        A(:, cols) = C(:, cols) + A(:, cols);
    end
    var_orig = var(reshape(A, [], num_epochs), 0, 1);
    A = [];
end
clear At
enova_per_epoch = zeros(1, num_epochs);
valid = var_orig > 0;
enova_per_epoch(valid) = var_art(valid) ./ var_orig(valid);
if num_epochs > 0
    state.ENOVA = mean(enova_per_epoch);
else
    state.ENOVA = 0;
end

state.C = C;
state.A = A;
state.SENSAI_score = SENSAI_score;
state.artifact_threshold_out = artifact_threshold_out;
state.samples_per_step = max(1, floor(256 * 2^20 / (4 * N_EEG_electrodes)));
state.next_sample = 1;
end

% =====================================================================
function [state, seg, first, last, artifacts] = local_legacy_step(state)
%LOCAL_LEGACY_STEP  Hand out the next slice of the band cleaned in 'begin'.
first = state.next_sample;
last  = min(state.P, first + state.samples_per_step - 1);
seg   = state.C(:, first:last);
if state.want_artifacts
    artifacts = state.A(:, first:last);
else
    artifacts = [];
end
state.next_sample = last + 1;
state.done = last >= state.P;
if state.done
    state.C = []; state.A = [];
end
end

% =====================================================================
function [Evec, Evald] = local_gevd(src, cls, off, K, R_chol, r, N, T)
%LOCAL_GEVD  Per-epoch generalized eigendecomposition against refCOV_reg, for
%   the K epochs of one grid (grid offset off), read in the same chunks the
%   epoched-array version used.
Evec  = zeros(N, r, K, cls);
Evald = zeros(r, K, cls);
if K == 0
    return
end

truncated = (r < N);
chunk_size = max(1, min(500, floor(2^23 / max(N * T, 1))));

for cs = 1:chunk_size:K
    ce = min(cs + chunk_size - 1, K);
    blk = reshape(gedai_read(src, off + (cs-1)*T + 1, off + ce*T, cls), N, T, []);
    blk = blk - mean(blk, 2);

    if truncated
        Z = R_chol' \ reshape(blk, N, []);
        Z = reshape(Z, N, T, []);
        for k = 1:(ce - cs + 1)
            [U, S, ~] = svd(Z(:,:,k), 'econ');
            s = diag(S);
            Evec(:,:,cs+k-1) = R_chol \ U(:,1:r);
            Evald(:,cs+k-1)  = s(1:r).^2 / (T - 1);
        end
    else
        C = pagemtimes(blk, 'none', blk, 'transpose') / (T - 1);
        for k = 1:(ce - cs + 1)
            A = C(:,:,k);
            A = (A + A') / 2;
            A_white = (R_chol' \ A) / R_chol;
            A_white = (A_white + A_white') / 2;
            [U, D] = eig(A_white);
            Evec(:,:,cs+k-1) = R_chol \ U;
            Evald(:,cs+k-1)  = diag(D);
        end
    end
end
end
