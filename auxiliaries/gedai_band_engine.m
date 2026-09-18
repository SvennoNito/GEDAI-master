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
%   [SENSAI_score, artifact_threshold_out, ENOVA, captured] = gedai_band_engine('finish', state);
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
%     .capture_windows   [nWin x 2] (sampleStart, sampleEnd), this band's own absolute
%                        1..P sample space (the same space 'step' reports first/last
%                        in), one row per window whose removed field the caller wants
%                        back - opt-in and windows-only (multivariate-prep issue #25,
%                        ADR 0004): passing zeros(0, 2) (the default) costs nothing,
%                        since the whole-segment/whole-band artifact arrays this would
%                        otherwise need are never built for want_artifacts = false (see
%                        local_capture_stream, local_capture_legacy). 'finish' returns
%                        one entry per row, in the same order, each an [N x winLen]
%                        matrix in the band's working precision ([] rows are never
%                        produced - every row must fall inside this band's [1, P]).
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
             'legacy_artifacts', true, 'thresh_plausibility_gate', 0, ...
             'thresh_window_min_epochs', 30, 'thresh_window_max_epochs', 200, ...
             'thresh_window_recalibrate', false, 'thresh_window_aggregate', 'mean', ...
             'capture_windows', zeros(0, 2));
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

% ---- optional plausibility floor -----------------------------------------
% A fixed, scale-free floor under the threshold: a component whose topography
% needs no more than opts.thresh_plausibility_gate leadfield principal
% components to reach 90% of its energy is never removed, whatever the
% threshold says. See local_npc90 (stream path) / clean_EEG_epoch (legacy
% path). Built once here, in the band's working precision, since both paths
% need it identically. gate = 0 (default) disables it and reproduces the
% previous behaviour exactly; it can only ever remove less, never more.
gate = opts.thresh_plausibility_gate;
if isempty(gate), gate = 0; end
if gate > 0
    [Ulf_, Dlf_] = eig(double(refCOV));
    [~, oLf] = sort(diag(Dlf_), 'descend');
    Ulf = cast(Ulf_(:, oLf), 'like', refCOV);
    clear Ulf_ Dlf_
else
    Ulf = [];
end

% The streaming path pays for itself only when there are many epochs. Its
% threshold stage decomposes up to 500 epochs for SENSAI and then sweeps the
% spectrum of every epoch; when the band has fewer than ~500 epochs those are
% the same epochs twice over, and the legacy path - one decomposition, kept -
% is simply the better algorithm. Long-epoch bands are exactly that case, and
% their stored eigenbasis is small because K = T/Te is small.
%
% A finite smoothing_window_seconds used to force the legacy path, which is what
% made the sliding threshold unusable on a whole night: legacy keeps Evec for
% every epoch (N x r x K), and on the fast bands of an 8 h recording that is tens
% of terabytes. local_windowed_threshold (below) optimises the threshold per
% window itself on the streaming path, so the sliding threshold and the
% streaming path are no longer exclusive.
epochs_in_band = floor(P / Te);
use_stream = ~opts.force_legacy && ...
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
state.gate = gate;
state.Ulf = Ulf;
state.smoothing_window_seconds = smoothing_window_seconds;
state.thresh_window_min_epochs = opts.thresh_window_min_epochs;
state.thresh_window_aggregate  = opts.thresh_window_aggregate;
%%% Removed-field capture (multivariate-prep issue #25, ADR 0004): a small,
%%% windows-sized accumulator per requested window, never a segment- or
%%% band-sized one. Empty by default (opts.capture_windows = zeros(0, 2)), so
%%% this loop does not run and production pays nothing.
state.capture_windows = opts.capture_windows;
state.captured = cell(size(state.capture_windows, 1), 1);
for iCap = 1:size(state.capture_windows, 1)
    wS = state.capture_windows(iCap, 1); wE = state.capture_windows(iCap, 2);
    state.captured{iCap} = zeros(N, wE - wS + 1, cls);
end

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
function [SENSAI_score, artifact_threshold_out, ENOVA, captured] = local_finish(state)
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
captured = state.captured;
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
epoch_size = state.epoch_size;
smoothing_window_seconds = state.smoothing_window_seconds;

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

% (A1) Exact global eigenvalue percentile, per grid, from an
% eigenvalues-only pass. Cheaper than the decomposition with vectors, and it
% reproduces what clean_EEG computed from the full stored spectrum.
% When the epoch is longer than the channel count there is no cheap
% eigenvalues-only route: eig(N x N) without vectors still costs ~60% of the
% full decomposition, so redoing it in the cleaning stage would be a net loss.
% In that regime the pre-pass keeps the leading eigenvectors as well, and the
% cleaning stage becomes pure application. Those bands have few epochs
% (K = T/Te), so the cache is small. For the short-epoch bands the opposite
% holds: the Gram route is cheap and K is large, so nothing is kept.
%
% This runs BEFORE the threshold is chosen: it does not depend on it, and the
% windowed optimiser below needs the global percentile in order to state its
% per-window thresholds on the same scale the cleaning stage applies them.
keep_vectors = ~truncated;
k_keep = min(r, 32);

[Evald_all_1, Utop_1, Dtop_1] = local_gevd_prepass(src, cls, 0, Te, R_chol, r, truncated, N, K, opts, keep_vectors, k_keep);
if K2 > 0
    [Evald_all_2, Utop_2, Dtop_2] = local_gevd_prepass(src, cls, sh, Te, R_chol, r, truncated, N, K2, opts, keep_vectors, k_keep);
else
    Evald_all_2 = []; Utop_2 = []; Dtop_2 = [];
end

% (A2) The artifact threshold t.
if ~isempty(opts.artifact_threshold_override)
    % Threshold supplied by the caller. Besides making the cleaning stage
    % testable in isolation, this is the only way to give two datasets the
    % same operating point: SENSAI is flat over a wide range, so letting it
    % re-optimise per run makes the amount removed depend on what else is in
    % the file.
    artifact_threshold = repmat(opts.artifact_threshold_override, 1, K);

elseif isinf(smoothing_window_seconds)
    % One threshold for the whole band. SENSAI subsample drawn exactly as
    % SENSAI_fminbnd would have drawn it from the full set, so the optimiser sees
    % the same epochs in the same order and returns the same threshold.
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
    artifact_threshold = repmat(artifact_threshold_scalar, 1, K);

else
    if ~(ischar(optimization_type) && strcmp(optimization_type, 'parabolic'))
        error('gedai_band_engine:optimization', ...
              'Only the ''parabolic'' optimizer is supported on the streaming path.');
    end
    artifact_threshold = local_windowed_threshold(src, cls, 0, K, Te, R_chol, r, ...
        truncated, N, refCOV, minThreshold, maxThreshold, noise_multiplier, ...
        evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold, ...
        epoch_size, smoothing_window_seconds, pct, Evald_all_1, opts);
end

artifact_threshold   = max(minThreshold, min(maxThreshold, artifact_threshold));
artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
if isempty(artifact_threshold_2), artifact_threshold_2 = artifact_threshold; end

T1_1 = (105 - artifact_threshold) / 100;
cut_1 = gedai_eig_threshold(Evald_all_1, N, pct, T1_1);
clear Evald_all_1

if K2 > 0
    T1_2 = (105 - artifact_threshold_2) / 100;
    cut_2 = gedai_eig_threshold(Evald_all_2, N, pct, T1_2);
    clear Evald_all_2
else
    cut_2 = [];
end
if opts.verbose
    fprintf('  [stream] threshold stage: %.2f s (t = %.2f .. %.2f, median %.2f)\n', ...
        toc(tA), min(artifact_threshold), max(artifact_threshold), median(artifact_threshold));
end

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
function t_per_epoch = local_windowed_threshold(src, cls, off, K, Te, R_chol, r, truncated, N, ...
    refCOV, minThreshold, maxThreshold, noise_multiplier, evecs_Template_cov, ...
    signal_type, SSI_top_PCs, percentile_threshold, epoch_size, ...
    smoothing_window_seconds, pct, Evald_all, opts)
%LOCAL_WINDOWED_THRESHOLD  One SENSAI optimum per sliding window, interpolated.
%
%   Why this exists. GEDAI's artifact criterion is an eigenvalue cut, and the cut
%   sits at exp(T1*(prctile(log(lambda)+100, 98)) - 100) with T1 = (105-t)/100.
%   Because the percentile is added to 100 before T1 scales it, one unit of t moves
%   the cut by a factor of e - and measured on a real night the share of a band that
%   goes from untouched to almost entirely removed spans about five units of t. So
%   the single scalar t that SENSAI picks for a recording is the most consequential
%   number in the whole method.
%
%   On a whole night that scalar is a compromise. The generalized eigenvalue spectra
%   are near identical across sleep stages (the leadfield whitening cancels the
%   delta-power difference), but the leading eigenVECTORS are not: in N3 they point
%   at slow waves, which are leadfield-plausible, and in wake at blinks and muscle,
%   which are not. SENSAI scores exactly that difference, so its optimum is genuinely
%   stage-dependent, and averaging it over a night lands between the two - aggressive
%   enough to attenuate slow waves, mild enough to leave wake artifacts.
%
%   Optimising per window instead keeps one uniform, stage-blind rule for the whole
%   recording while letting the operating point follow the data, including within a
%   stage (a quiet N2 stretch and one full of arousals are not the same problem).
%
%   The windowing, smoothing and makima interpolation deliberately mirror
%   local_legacy_begin, so the sliding threshold means the same thing on both paths.

nSub = min(opts.thresh_window_max_epochs, K);

%%% Window geometry. A window is given in seconds, but SENSAI needs a decent number
%%% of epochs to be stable and the slow bands have epochs tens of seconds long, so
%%% the epoch floor wins there and those bands simply use longer windows.
win = max(1, round(smoothing_window_seconds / epoch_size));
win = min(max(win, opts.thresh_window_min_epochs), K);
step = max(1, round(win / 2));
if K <= win
    nwin = 1; win = K;
else
    nwin = ceil((K - win) / step) + 1;
end

centers = zeros(1, nwin);
widx    = cell(1, nwin);
for w = 1:nwin
    i0 = (w-1)*step + 1;
    i1 = min(K, i0 + win - 1);
    if w == nwin && (i1 - i0 + 1) < win/2 && nwin > 1
        i0 = max(1, K - win + 1); i1 = K;
    end
    centers(w) = (i0 + i1) / 2;
    ii = i0:i1;
    if numel(ii) > nSub
        ii = ii(round(linspace(1, numel(ii), nSub)));
    end
    widx{w} = ii;
end

%%% Per-window optimisation. Windows are independent, so they go to the pool in
%%% waves; as everywhere else here the slices are cut on the client, because a
%%% parfor cannot slice a source on a window's epoch list and would broadcast the band.
topt = zeros(1, nwin);
if opts.parallel_blocks && nwin > 1
    p = gcp('nocreate');
    if isempty(p), nw = 1; else, nw = p.NumWorkers; end
    bytes_per_win = N * Te * nSub * 8;
    wave = max(1, min(2*nw, floor(32*2^30 / max(bytes_per_win, 1))));
else
    wave = 1;
end

for w0 = 1:wave:nwin
    w1 = min(nwin, w0 + wave - 1);
    m  = w1 - w0 + 1;
    Xw = cell(1, m);
    for j = 1:m
        ii = widx{w0+j-1};
        Xj = zeros(N, Te, numel(ii), cls);
        for q = 1:numel(ii)
            s0 = off + (ii(q)-1)*Te;
            Xj(:,:,q) = gedai_read(src, s0+1, s0+Te, cls);
        end
        Xw{j} = Xj;
    end
    tj = zeros(1, m);
    if wave > 1
        parfor j = 1:m
            [Ev, Ed] = local_gevd_subset_core(Xw{j}, Te, R_chol, r, truncated, N);
            tj(j) = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Ed, Ev, ...
                noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, ...
                percentile_threshold);
        end
    else
        for j = 1:m
            [Ev, Ed] = local_gevd_subset_core(Xw{j}, Te, R_chol, r, truncated, N);
            tj(j) = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Ed, Ev, ...
                noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, ...
                percentile_threshold);
        end
    end
    topt(w0:w1) = tj;
    clear Xw
end

%%% Optionally restate each window's threshold against the global eigenvalue percentile.
%%%
%%% SENSAI evaluates t on its window's own spectrum, but the cleaning stage applies it
%%% against the percentile pooled over the whole band, so the cut that lands is not the
%%% cut SENSAI chose - it is off by exp(T1*(Lp_global - Lp_local)). That looked like a
%%% defect worth correcting, and this is the correction.
%%%
%%% Measured, it is not. Wake windows sit 2-3 log units above the night's pooled
%%% percentile (the artifact load is genuinely larger there, unlike the three sleep
%%% stages, which agree to within ~0.5), so leaving the drift in place makes the applied
%%% cut harsher exactly where the window is noisy and gentler where it is clean - a
%%% second, automatic adaptation on top of the window's own t. On one night, disabling
%%% the correction left slow-wave preservation unchanged (10th percentile 99.3 % vs
%%% 99.0 %) while removing meaningfully more wake artifact (wake EMG 3.35 vs 5.08,
%%% wake delta 20.2 vs 24.2 uV^2). So the default is off, which is also GEDAI's original
%%% behaviour; the option stays for anyone who wants the applied cut to be exactly the
%%% chosen one.
if opts.thresh_window_recalibrate
    Lp_glob = local_pool_percentile(Evald_all, N, pct);
    for w = 1:nwin
        Lp_loc = local_pool_percentile(Evald_all(:, widx{w}), N, pct);
        if isfinite(Lp_loc) && isfinite(Lp_glob) && Lp_glob ~= 0
            T1w  = (105 - topt(w)) / 100;
            topt(w) = 105 - 100 * (T1w * Lp_loc / Lp_glob);
        end
    end
end

topt = max(minThreshold, min(maxThreshold, topt));

%%% How neighbouring windows are combined.
%%%
%%% 'mean' is GEDAI's original behaviour: a 3-window moving average, which with 50 %
%%% overlap spreads each window's influence over about two window lengths. That is
%%% fine on average and wrong exactly at a stage transition, where it produces a
%%% smooth RAMP of cleaning strength across the boundary. A ramp is the worst shape
%%% for an analysis of the transition itself, because it is indistinguishable from a
%%% physiological gradient in the thing being measured.
%%%
%%% 'min' takes the gentlest of each window's neighbours instead. Higher t is more
%%% aggressive, so the minimum lets the sleep side's operating point extend into the
%%% first wake window rather than the wake side's reaching back into sleep. It is the
%%% continuous, data-driven analogue of gedai.dilateStages, and it makes the error at
%%% a boundary one-sided: under-cleaned wake, which is visible in the data and can be
%%% handled downstream, rather than removed slow waves, which cannot be recovered.
if nwin >= 3
    switch lower(opts.thresh_window_aggregate)
        case 'mean', topt = smoothdata(topt, 'movmean', 3);
        case 'min',  topt = movmin(topt, 3);
        otherwise
            error('gedai_band_engine:aggregate', ...
                'thresh_window_aggregate must be ''mean'' or ''min''.');
    end
end

if nwin > 1
    padded_centers    = [1, centers, K];
    padded_thresholds = [topt(1), topt, topt(end)];
    [uc, ui] = unique(padded_centers);
    ut = padded_thresholds(ui);
    t_per_epoch = interp1(uc, ut, 1:K, 'makima');
    %%% makima can overshoot its data, and an overshoot upward is an overshoot towards
    %%% MORE cleaning than any neighbouring window asked for. Under 'mean' that is
    %%% legacy behaviour and left alone; under 'min' it would silently defeat the whole
    %%% point at exactly the stage boundaries the option exists to protect, so the
    %%% interpolant is clamped to the envelope of the two nodes bracketing each epoch.
    %%% That makes the guarantee structural rather than a property of the spline.
    if strcmpi(opts.thresh_window_aggregate, 'min')
        lo = interp1(uc, ut, 1:K, 'previous');
        hi = interp1(uc, ut, 1:K, 'next');
        lo(isnan(lo)) = ut(1);
        hi(isnan(hi)) = ut(end);
        t_per_epoch = min(max(t_per_epoch, min(lo, hi)), max(lo, hi));
    end
else
    t_per_epoch = repmat(topt, 1, K);
end
end

% =====================================================================
function Lp = local_pool_percentile(evals, num_chans, pct)
%LOCAL_POOL_PERCENTILE  The shifted-log percentile gedai_eig_threshold works from,
%   including its null-space remap, exposed on its own so a window's threshold can
%   be restated against a different pool.
mag = abs(evals);
lv  = log(mag(mag > 0)) + 100;
if isempty(lv), Lp = NaN; return; end
n_null = max(0, num_chans - size(evals, 1)) * size(evals, 2);
if n_null > 0
    pe = min(100, max(0, (pct/100 * (numel(lv) + n_null) - n_null) / numel(lv) * 100));
else
    pe = pct;
end
Lp = prctile(lv, pe);
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
Ulf = state.Ulf; gate = state.gate;

pj = find(onPool);
if ~isempty(pj)
    pX = Xb(pj); pU = Ub(pj); pD = Db(pj); pcs = css(pj); pce = ces(pj); pg = grid(pj);
    m = numel(pj);
    pseg = cell(1, m); psg = cell(1, m); png = cell(1, m);
    parfor j = 1:m
        if pg(j) == 1
            [pseg{j}, psg{j}, png{j}] = local_clean_block(pX{j}, pcs(j), pce(j), K, Te, R_chol, B, r, ...
                truncated, N, cut_1, cw, true, Template_guess, T_proj, M_ssi, pU{j}, pD{j}, Ulf, gate);
        else
            [pseg{j}, psg{j}, png{j}] = local_clean_block(pX{j}, pcs(j), pce(j), K2, Te, R_chol, B, r, ...
                truncated, N, cut_2, cw, false, [], [], M_ssi, pU{j}, pD{j}, Ulf, gate);
        end
    end
    segs(pj) = pseg; sgs(pj) = psg; ngs(pj) = png;
    clear pX pU pD pseg
end
for j = find(~onPool)
    if grid(j) == 1
        [segs{j}, sgs{j}, ngs{j}] = local_clean_block(Xb{j}, css(j), ces(j), K, Te, R_chol, B, r, ...
            truncated, N, cut_1, cw, true, Template_guess, T_proj, M_ssi, Ub{j}, Db{j}, Ulf, gate);
    else
        [segs{j}, sgs{j}, ngs{j}] = local_clean_block(Xb{j}, css(j), ces(j), K2, Te, R_chol, B, r, ...
            truncated, N, cut_2, cw, false, [], [], M_ssi, Ub{j}, Db{j}, Ulf, gate);
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

if ~isempty(state.capture_windows)
    state = local_capture_stream(state, seg, first, last, cls);
end

state.next_block = b1 + 1;
state.done = b1 >= state.nblocks_1;
if state.done
    state.pend_O = []; state.pend_C = []; state.src = [];
    state.Utop_1 = []; state.Dtop_1 = []; state.Utop_2 = []; state.Dtop_2 = [];
end
end

% =====================================================================
function state = local_capture_stream(state, seg, first, last, cls)
%LOCAL_CAPTURE_STREAM  Windows-only removed-field capture (multivariate-prep
%   issue #25, ADR 0004). For each caller-supplied window overlapping this
%   step's [first, last], read just the overlapping raw samples and subtract
%   the matching columns of the already-computed cleaned segment - the same
%   arithmetic as the want_artifacts branch above, restricted to the tiny
%   overlap instead of the whole segment. state.src is still valid here (it
%   is only released once state.done), and nothing band- or segment-sized is
%   ever held: state.captured{iCap} is exactly the requested window's width.
CW = state.capture_windows;
for iCap = 1:size(CW, 1)
    wS = CW(iCap, 1); wE = CW(iCap, 2);
    ov0 = max(wS, first); ov1 = min(wE, last);
    if ov0 > ov1, continue; end
    raw = gedai_read(state.src, ov0, ov1, cls);
    cleaned = seg(:, ov0 - first + 1 : ov1 - first + 1);
    state.captured{iCap}(:, ov0 - wS + 1 : ov1 - wS + 1) = raw - cleaned;
end
end

% =====================================================================
function [seg, sig_b, noi_b, nb_sum, nb_max] = local_clean_block( ...
    Xblk, cs, ce, K, Te, R_chol, B, r, truncated, N, cut, cosine_weights, ...
    do_sensai, Template_guess, T_proj, M_ssi, Utop, Dtop, Ulf, gate)
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

    Xk = D3(:,:,k);
    if ~isempty(bad)
        V_bad = R_chol \ U;               % back-transform only what is removed
        d_bad = abs(d_all(bad));
        d_bad = d_bad(:);
        if gate > 0
            %%% Plausibility floor. The threshold is an amplitude criterion, and
            %%% amplitude does not separate brain from artifact - a slow wave and a
            %%% movement artifact are both large. That is why the threshold has to be
            %%% adapted at all, and adapting it is what makes the same eigenvalue
            %%% removable in one window and not another. This is the one part of the
            %%% decision that is scale-free, and therefore means the same thing
            %%% everywhere: a component whose topography lives in a handful of leadfield
            %%% directions is a dipolar source and is kept, whatever the threshold says.
            %%%
            %%% Measured on one night: slow waves need 7 leadfield PCs, K-complexes 5,
            %%% spindles 5, wake alpha 8, while the components GEDAI removes in wake need
            %%% ~180. At a gate of 15 nothing GEDAI removes in wake is released.
            %%% It cannot REPLACE the threshold - once both populations are above the cut
            %%% they overlap badly (AUC ~0.75) - so it is a floor only, and by
            %%% construction it can remove less, never more.
            keepBad = local_npc90(B, V_bad, Ulf) > gate;
            V_bad = V_bad(:, keepBad);
            % Indexing a single-element vector with an all-false logical collapses
            % to 0x0 rather than 0x1 (a MATLAB quirk), which breaks the (num_bad x
            % M_ssi) broadcast in local_sensai_epoch below. Force the column shape.
            d_bad = reshape(d_bad(keepBad(:)), [], 1);
        end
    else
        V_bad = zeros(N, 0, 'like', Xblk); d_bad = zeros(0, 1, 'like', Xblk);
    end

    num_bad = size(V_bad, 2);
    nb_sum  = nb_sum + num_bad;
    nb_max  = max(nb_max, num_bad);
    if num_bad > 0
        if num_bad <= Te
            Xk = Xk - (B * V_bad) * (V_bad' * Xk);
        else
            Xk = Xk - B * (V_bad * (V_bad' * Xk));
        end
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
function npc = local_npc90(B, V, Ulf)
%LOCAL_NPC90  How dipolar each component is, on a scale that does not depend on its size.
%
%   For generalized eigenvector v the activation pattern - the map that actually gets
%   subtracted - is a = B*v. Expanded in the eigenbasis of the leadfield gram, a smooth
%   dipolar field concentrates in the leading directions while a single-channel pop or a
%   muscle burst spreads thinly over all of them. Returned is the number of leadfield
%   principal components needed to reach 90 % of the map's energy: invariant to the
%   component's amplitude, and therefore the same criterion in every window and stage.
A = B * V;
nrm = sqrt(sum(A.^2, 1)); nrm(nrm == 0) = 1;
A = A ./ nrm;
E = (Ulf' * A).^2;
C = cumsum(E, 1) ./ max(sum(E, 1), realmin);
npc = sum(C < 0.90, 1) + 1;
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
Ulf = state.Ulf; gate = state.gate;
thresh_window_min_epochs = state.thresh_window_min_epochs;
thresh_window_aggregate  = state.thresh_window_aggregate;

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
%%% A window is specified in seconds, but SENSAI optimises over epochs and needs
%%% enough of them to be stable. The slowest wavelet bands have epochs of one to
%%% three minutes, where a 300 s window holds two or three - not a distribution.
%%% Those bands therefore use a longer window than asked for, rather than a
%%% threshold fitted to a handful of epochs. Same floor as local_windowed_threshold.
if ~isinf(smoothing_window_seconds)
    window_epochs = min(max(window_epochs, thresh_window_min_epochs), N_epochs);
end
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
        %%% See local_windowed_threshold: 'min' keeps a stage transition from pulling a
        %%% wake-strength threshold back over sleep, at the cost of under-cleaning the
        %%% first window after the transition.
        switch lower(thresh_window_aggregate)
            case 'mean', optimal_threshold_per_window = smoothdata(optimal_threshold_per_window, 'movmean', 3);
            case 'min',  optimal_threshold_per_window = movmin(optimal_threshold_per_window, 3);
            otherwise
                error('gedai_band_engine:aggregate', ...
                    'thresh_window_aggregate must be ''mean'' or ''min''.');
        end
    end
    padded_centers    = [1, window_centers, N_epochs];
    padded_thresholds = [optimal_threshold_per_window(1), optimal_threshold_per_window, optimal_threshold_per_window(end)];
    [unique_centers, unique_idx] = unique(padded_centers);
    unique_thresholds = padded_thresholds(unique_idx);
    artifact_threshold_array = interp1(unique_centers, unique_thresholds, 1:N_epochs, 'makima');
    %%% makima can overshoot its data - see local_windowed_threshold for why that is
    %%% only acceptable under 'mean'. Under 'min' the interpolant is clamped to the
    %%% envelope of the two nodes bracketing each epoch.
    if strcmpi(thresh_window_aggregate, 'min')
        lo = interp1(unique_centers, unique_thresholds, 1:N_epochs, 'previous');
        hi = interp1(unique_centers, unique_thresholds, 1:N_epochs, 'next');
        lo(isnan(lo)) = unique_thresholds(1);
        hi(isnan(hi)) = unique_thresholds(end);
        artifact_threshold_array = min(max(artifact_threshold_array, min(lo, hi)), max(lo, hi));
    end
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
%%% Capture windows (multivariate-prep issue #25, ADR 0004) that reach past len
%%% into the sub-epoch tail need At computed too, even when the caller did not
%%% ask for want_artifacts; capture_windows is empty by default, so this never
%%% fires in production.
if size(state.capture_windows, 1) > 0 && any(state.capture_windows(:, 2) > len)
    keep_tail = true;
end
C  = zeros(N_EEG_electrodes, P, cls);
A  = zeros(N_EEG_electrodes, len, cls);                 % artifacts of the whole epochs
At = zeros(N_EEG_electrodes, (P - len) * keep_tail, cls); % the rest, only for the caller

[thr_1, artifact_threshold_out, ~, mag_1] = clean_EEG_thresholds( ...
    Evald, N_EEG_electrodes, artifact_threshold, refCOV, signal_type, refCOV_reg, percentile_threshold);
for i = 1:N_epochs
    s0 = (i-1) * epoch_samples;
    X  = gedai_read(src, s0 + 1, s0 + epoch_samples, cls);
    [cl, ar] = clean_EEG_epoch(X, i, N_epochs, mag_1, thr_1, Evec, refCOV_reg, cosine_weights, epoch_samples, Ulf, gate);
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
    [cl, ar] = clean_EEG_epoch(X, i, K2, mag_2, thr_2, Evec_2, refCOV_reg, cosine_weights, epoch_samples, Ulf, gate);
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
%%% Removed-field capture (issue #25, ADR 0004): A (whole epochs) and At (the
%%% tail, populated above whenever a window needs it) still hold pure removed
%%% field here, before the branch below either concatenates or overwrites
%%% them in place. Reading it now means capture never needs its own copy of
%%% either array.
if size(state.capture_windows, 1) > 0
    state = local_capture_legacy(state, A, At, len);
end
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
function state = local_capture_legacy(state, A, At, len)
%LOCAL_CAPTURE_LEGACY  Windows-only removed-field capture (multivariate-prep
%   issue #25, ADR 0004). A (samples 1..len, the whole-epoch region) and At
%   (samples len+1..P, the sub-epoch tail - only populated when keep_tail was
%   forced on for a window reaching that far) hold this band's pure removed
%   field at the point local_legacy_begin calls this, before A is either
%   concatenated with At (want_artifacts) or overwritten in place with
%   cleaned + artifacts (~want_artifacts). Called only when capture_windows is
%   non-empty, so production (capture_windows = zeros(0, 2)) never runs it.
CW = state.capture_windows;
P = len + size(At, 2);
for iCap = 1:size(CW, 1)
    wS = CW(iCap, 1); wE = CW(iCap, 2);
    lo = max(wS, 1); hi = min(wE, P);
    if lo > hi, continue; end
    out = state.captured{iCap};
    a0 = max(lo, 1); a1 = min(hi, len);
    if a0 <= a1
        out(:, a0 - wS + 1 : a1 - wS + 1) = A(:, a0:a1);
    end
    t0 = max(lo, len + 1); t1 = min(hi, P);
    if t0 <= t1
        out(:, t0 - wS + 1 : t1 - wS + 1) = At(:, t0 - len : t1 - len);
    end
    state.captured{iCap} = out;
end
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
