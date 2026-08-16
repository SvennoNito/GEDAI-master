function [cleaned_data, SENSAI_score, artifact_threshold_out, ENOVA, diag_out] = ...
    gedai_band_stream(eeg_data, srate, epoch_size, refCOV, artifact_threshold_type, ...
                      optimization_type, signal_type, minThreshold, maxThreshold, ...
                      percentile_threshold, rank_truncation, opts)
%GEDAI_BAND_STREAM  Streaming implementation of one GEDAI band pass.
%
%   Same arithmetic as the loop in GEDAI_per_band, reorganised so that the
%   eigenvectors are never held for more than one block of epochs at a time.
%
%   The reorganisation rests on two facts about the method:
%
%   1. Only the components ABOVE the threshold are ever used. The cut sits at
%      lambda_98^T1 * exp(5 - t), so in practice a handful of the 47..256
%      components per epoch. Computing and storing the whole eigenbasis for
%      every epoch, as the previous version did, is ~99% wasted.
%
%   2. The signal covariance never has to be formed. Because the generalized
%      eigenvectors are B-orthonormal, COV = B*V*diag(d)*V'*B over the FULL
%      basis, so the "good" half is just COV minus the "bad" half. SENSAI
%      therefore needs the same handful of vectors the cleaning needs, and
%      COV applied as X*(X'*M)/(T-1).
%
%   The threshold path is deliberately left bit-identical to the previous
%   implementation: the SENSAI subsample is drawn with the same stream and the
%   same call, and the global eigenvalue percentile is taken over every epoch
%   via a cheap eigenvalues-only pre-pass rather than estimated from a sample.
%   The SENSAI objective is flat over a wide range of thresholds while the
%   amount of data removed across that same range is not, so an approximate
%   threshold is not a safe trade here.
%
%   opts fields (all optional)
%     .block_epochs   epochs per streaming block (default: sized to ~256 MB)
%     .parallel_blocks  run the blocks on a parallel pool (default false)
%     .verbose        print stage timings (default false)

N  = size(eeg_data, 1);
Te = round(srate * epoch_size);
if nargin < 12 || isempty(opts), opts = struct; end
def = struct('block_epochs', [], 'parallel_blocks', false, 'verbose', false, ...
             'artifact_threshold_override', []);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i})), opts.(fn{i}) = def.(fn{i}); end
end

K   = size(eeg_data, 2) / Te;      % stream-1 epochs (data is pre-padded)
K2  = K - 1;                       % stream-2 epochs
sh  = Te / 2;                      % stream 2 is offset by half an epoch

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
    [Evec_s, Evald_s] = local_gevd_subset(eeg_data, sensai_idx, 0, Te, R_chol, r, truncated, N, opts.parallel_blocks);

    switch optimization_type
        case 'parabolic'
            artifact_threshold_scalar = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, ...
                Evald_s, Evec_s, noise_multiplier, evecs_Template_cov, signal_type, ...
                SSI_top_PCs, percentile_threshold);
        otherwise
            error('gedai_band_stream:optimization', ...
                  'Only the ''parabolic'' optimizer is supported on the streaming path.');
    end
    clear Evec_s Evald_s
end

artifact_threshold_scalar = max(minThreshold, min(maxThreshold, artifact_threshold_scalar));
artifact_threshold   = repmat(artifact_threshold_scalar, 1, K);
artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
if isempty(artifact_threshold_2), artifact_threshold_2 = artifact_threshold; end

% (A2) Exact global eigenvalue percentile, per stream, from an
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

[Evald_all_1, Utop_1, Dtop_1] = local_gevd_prepass(eeg_data, 0, Te, R_chol, r, truncated, N, K, opts, keep_vectors, k_keep);
T1_1 = (105 - artifact_threshold) / 100;
cut_1 = gedai_eig_threshold(Evald_all_1, N, pct, T1_1);
clear Evald_all_1

if K2 > 0
    [Evald_all_2, Utop_2, Dtop_2] = local_gevd_prepass(eeg_data, sh, Te, R_chol, r, truncated, N, K2, opts, keep_vectors, k_keep);
    T1_2 = (105 - artifact_threshold_2) / 100;
    cut_2 = gedai_eig_threshold(Evald_all_2, N, pct, T1_2);
    clear Evald_all_2
else
    cut_2 = []; Utop_2 = []; Dtop_2 = [];
end
if opts.verbose, fprintf('  [stream] threshold stage: %.2f s\n', toc(tA)); end

%% ================= STAGE B : cleaning ===================================
tB = tic;
cosine_weights = create_cosine_weights(N, srate, epoch_size, 1);

% SENSAI accumulators (stream 1 only, matching the previous implementation)
M_ssi  = min(size(evecs_Template_cov, 2), SSI_top_PCs);
Template_guess = evecs_Template_cov(:, 1:M_ssi);
T_proj = refCOV_reg * Template_guess;

[cleaned_1, sig_dist, noi_dist, nbad_1] = local_clean_stream( ...
    eeg_data, 0, K, Te, R_chol, refCOV_reg, r, truncated, N, cut_1, cosine_weights, ...
    true, Template_guess, T_proj, M_ssi, Utop_1, Dtop_1, opts);
clear Utop_1 Dtop_1

if K2 > 0
    [cleaned_2, ~, ~, nbad_2] = local_clean_stream( ...
        eeg_data, sh, K2, Te, R_chol, refCOV_reg, r, truncated, N, cut_2, cosine_weights, ...
        false, [], [], M_ssi, Utop_2, Dtop_2, opts);
    clear Utop_2 Dtop_2

    % Edge weighting of the shifted stream, then overlap-add
    n2 = size(cleaned_2, 2); e2 = n2 - sh;
    cleaned_2(:, 1:sh)     = cleaned_2(:, 1:sh)     .* cosine_weights(:, 1:sh);
    cleaned_2(:, e2+1:end) = cleaned_2(:, e2+1:end) .* cosine_weights(:, sh+1:end);
    cleaned_1(:, sh+1:sh+n2) = cleaned_1(:, sh+1:sh+n2) + cleaned_2;
    clear cleaned_2
else
    nbad_2 = struct('sum', 0, 'max', 0, 'n', 0);
end
cleaned_data = cleaned_1;
clear cleaned_1
if opts.verbose, fprintf('  [stream] cleaning stage: %.2f s\n', toc(tB)); end

%% ---- SENSAI score, same definition as before ---------------------------
SIGNAL_subspace_similarity = 100 * mean(sig_dist);
NOISE_subspace_similarity  = 100 * mean(noi_dist);
SENSAI_score = SIGNAL_subspace_similarity - (noise_multiplier * NOISE_subspace_similarity);

artifact_threshold_out = artifact_threshold;
ENOVA = [];   % computed by the caller from eeg_data - cleaned_data
diag_out = struct('nbad_mean', (nbad_1.sum + nbad_2.sum) / max(1, nbad_1.n + nbad_2.n), ...
                  'nbad_max',  max(nbad_1.max, nbad_2.max), ...
                  'rank', r, 'cut_stream1', cut_1(1));
end

% =====================================================================
function [cleaned, sig_dist, noi_dist, nbad] = local_clean_stream( ...
    X, off, K, Te, R_chol, B, r, truncated, N, cut, cosine_weights, ...
    do_sensai, Template_guess, T_proj, M_ssi, Utop, Dtop, opts)
%LOCAL_CLEAN_STREAM  Clean one epoch grid, block by block.

block = opts.block_epochs;
if isempty(block)
    % keep the per-block working set near 256 MB regardless of band
    bytes = 8; if isa(X, 'single'), bytes = 4; end
    block = max(1, min(512, floor(256*2^20 / max(6 * N * Te * bytes, 1))));
end
nblocks = ceil(K / block);

cleaned  = zeros(N, K*Te, 'like', X);
sig_dist = zeros(1, K);
noi_dist = zeros(1, K);
nbad = struct('sum', 0, 'max', 0, 'n', K);

if opts.parallel_blocks && nblocks > 1
    % Blocks are independent once the threshold is fixed, so they go to a pool.
    % They are dispatched in waves rather than all at once: a parfor cannot
    % slice X (the block offset is not the loop variable), so each wave's
    % blocks are cut out first and only those slices cross to the workers.
    % That keeps the extra allocation at wave_size x block instead of a full
    % broadcast copy of the band per worker.
    p = gcp('nocreate');
    if isempty(p), nw = 0; else, nw = p.NumWorkers; end
    wave = max(1, 2 * max(nw, 1));

    for w0 = 1:wave:nblocks
        w1 = min(nblocks, w0 + wave - 1);
        m  = w1 - w0 + 1;
        Xb = cell(1, m); Ub = cell(1, m); Db = cell(1, m);
        css = zeros(1, m); ces = zeros(1, m);
        for j = 1:m
            b = w0 + j - 1;
            css(j) = (b-1)*block + 1;
            ces(j) = min(K, b*block);
            s0 = off + (css(j)-1)*Te;
            Xb{j} = X(:, s0+1 : s0 + (ces(j)-css(j)+1)*Te);
            if ~isempty(Utop)
                Ub{j} = Utop(:,:,css(j):ces(j));
                Db{j} = Dtop(:,css(j):ces(j));
            end
        end
        segs = cell(1, m); sgs = cell(1, m); ngs = cell(1, m);
        nbs = zeros(1, m); nbm = zeros(1, m);
        parfor j = 1:m
            [segs{j}, sgs{j}, ngs{j}, nbs(j), nbm(j)] = local_clean_block( ...
                Xb{j}, css(j), ces(j), K, Te, R_chol, B, r, truncated, N, cut, ...
                cosine_weights, do_sensai, Template_guess, T_proj, M_ssi, Ub{j}, Db{j});
        end
        for j = 1:m
            cleaned(:, (css(j)-1)*Te+1 : ces(j)*Te) = segs{j};
            if do_sensai
                sig_dist(css(j):ces(j)) = sgs{j};
                noi_dist(css(j):ces(j)) = ngs{j};
            end
            nbad.sum = nbad.sum + nbs(j);
            nbad.max = max(nbad.max, nbm(j));
        end
    end
else
    for b = 1:nblocks
        cs = (b-1)*block + 1;
        ce = min(K, b*block);
        if isempty(Utop), Ub = []; Db = []; else, Ub = Utop(:,:,cs:ce); Db = Dtop(:,cs:ce); end
        s0 = off + (cs-1)*Te;
        Xblk = X(:, s0+1 : s0 + (ce-cs+1)*Te);
        [seg, s_b, n_b, nb_sum, nb_max] = local_clean_block( ...
            Xblk, cs, ce, K, Te, R_chol, B, r, truncated, N, cut, cosine_weights, ...
            do_sensai, Template_guess, T_proj, M_ssi, Ub, Db);
        cleaned(:, (cs-1)*Te+1 : ce*Te) = seg;
        if do_sensai
            sig_dist(cs:ce) = s_b;
            noi_dist(cs:ce) = n_b;
        end
        nbad.sum = nbad.sum + nb_sum;
        nbad.max = max(nbad.max, nb_max);
    end
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
    g = cs + k - 1;                       % global epoch index in this stream
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
%   The previous code applied  cov_signal * Y = B*V_good*diag(d_good)*V_good'*Y.
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
function [Evec, Evald] = local_gevd_subset(X, idx, off, Te, R_chol, r, truncated, N, parallel_blocks)
%LOCAL_GEVD_SUBSET  Full decomposition, for the SENSAI epochs only. The
%   optimiser sweeps thresholds, so the good/bad split moves and it genuinely
%   needs the whole eigenbasis on these few hundred epochs. Several seconds a
%   band, so it goes to the pool as well when one is in use.
if nargin < 9, parallel_blocks = false; end
n = numel(idx);
Xs = zeros(N, Te, n, 'like', X);
for j = 1:n
    s0 = off + (idx(j)-1)*Te;
    Xs(:,:,j) = X(:, s0+1 : s0+Te);
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
    Ec = cell(1, m); Dc = cell(1, m);
    parfor j = 1:m
        [Ec{j}, Dc{j}] = local_gevd_subset_core(Xc{j}, Te, R_chol, r, truncated, N);
    end
    Evec  = zeros(N, r, n, 'like', X);
    Evald = zeros(r, n, 'like', X);
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
function [Evald, Utop, Dtop] = local_gevd_prepass(X, off, Te, R_chol, r, truncated, N, K, opts, keep_vectors, k_keep)
%LOCAL_GEVD_PREPASS  Spectrum of every epoch, so that the global percentile is
%   exact rather than sampled. Optionally also keeps the leading k_keep
%   whitened eigenvectors, for the regime where recomputing them in the
%   cleaning stage would cost more than storing them.
Evald = zeros(r, K, 'like', X);
if keep_vectors
    Utop = zeros(N, k_keep, K, 'like', X);
    Dtop = zeros(k_keep, K, 'like', X);
else
    Utop = []; Dtop = [];
end
block = opts.block_epochs;
if isempty(block)
    bytes = 8; if isa(X, 'single'), bytes = 4; end
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
            Xb{j} = X(:, s0+1 : s0 + (ces(j)-css(j)+1)*Te);
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
        Xblk = X(:, s0+1 : s0 + (ce-cs+1)*Te);
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
