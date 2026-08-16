% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold, smoothing_window_seconds, percentile_threshold, rank_truncation, opts)
%   rank_truncation - logical, default true. When an epoch is shorter than the
%                     channel count its covariance is rank deficient, and the
%                     GEVD is truncated to the supported subspace. This is the
%                     large memory and speed win on the fast bands. Set false
%                     to force the full N x N decomposition, which is the
%                     reference behaviour; see the note at local_gevd.
%
%   opts            - struct of streaming options, see gedai_band_stream.
%                     .want_artifacts   return artifacts_data (default true).
%                                       GEDAI.m discards it, and it is a full
%                                       extra copy of the band, so the band
%                                       loop passes false.
%                     .block_epochs     epochs per streaming block.
%                     .parallel_blocks  run blocks on a parallel pool.
%                     .force_legacy     use the pre-streaming implementation.
%
%   Two implementations live here. The streaming one (default whenever
%   smoothing_window_seconds is Inf, which is the global-threshold case) never
%   materialises the eigenvectors for the whole band. The legacy one is kept
%   verbatim for the sliding-window path and as a reference oracle.

if isempty(eeg_data)
    error('Cannot process empty data');
end
if ~ismatrix(eeg_data)
    error('Input EEG data must be a 2D matrix (channels x samples).');
end
N_EEG_electrodes = size(eeg_data, 1);
if ~isa(eeg_data, 'double') && ~isa(eeg_data, 'single')
    eeg_data = double(eeg_data); % Only cast if not already float
end

% Ensure refCOV matches precision of eeg_data
refCOV = cast(refCOV, 'like', eeg_data);
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;

if nargin < 9  || isempty(signal_type),  signal_type  = 'eeg'; end
if nargin < 10 || isempty(minThreshold), minThreshold = 0;     end
if nargin < 11 || isempty(maxThreshold), maxThreshold = 12;    end
if nargin < 12 || isempty(smoothing_window_seconds), smoothing_window_seconds = Inf; end
if nargin < 13, percentile_threshold = []; end
if nargin < 14 || isempty(rank_truncation), rank_truncation = true; end
if nargin < 15 || isempty(opts), opts = struct; end
if ~isfield(opts, 'want_artifacts'), opts.want_artifacts = true;  end
if ~isfield(opts, 'force_legacy'),   opts.force_legacy   = false; end
if ~isfield(opts, 'artifact_threshold_override'), opts.artifact_threshold_override = []; end
if ~isfield(opts, 'precision'),      opts.precision      = 'double'; end

% ---- precision policy ---------------------------------------------------
% 'auto' runs a band in single precision only when its epoch is longer than the
% channel count. Measured on a real night, 256 ch:
%   epoch 384 samples : 1.97x faster, cleaned data differs by 1.3e-5 relative
%   epoch  48 samples : 1.21x faster, cleaned data differs by 3.3e-2 relative
% The second number is not rounding noise. Short epochs go through the Gram
% matrix of the whitened epoch, which squares the condition number, and the
% components that move are the ones sitting near a threshold that SENSAI only
% weakly determines. So the split is deliberate, not a tuning constant.
epoch_samples_for_precision = round(srate * epoch_size);
switch lower(opts.precision)
    case 'single', use_single = true;
    case 'auto',   use_single = epoch_samples_for_precision > N_EEG_electrodes;
    otherwise,     use_single = false;
end
cast_back = false;
if use_single && isa(eeg_data, 'double')
    eeg_data  = single(eeg_data);
    refCOV    = single(refCOV);
    cast_back = true;   % the caller accumulates bands in double
end

% The streaming path pays for itself only when there are many epochs. Its
% threshold stage decomposes up to 500 epochs for SENSAI and then sweeps the
% spectrum of every epoch; when the band has fewer than ~500 epochs those are
% the same epochs twice over, and the legacy path - one decomposition, kept -
% is simply the better algorithm. Long-epoch bands are exactly that case, and
% their stored eigenbasis is small because K = T/Te is small.
epochs_in_band = floor(size(eeg_data, 2) / round(srate * epoch_size));
use_stream = isinf(smoothing_window_seconds) && ~opts.force_legacy && ...
             (ischar(optimization_type) && strcmp(optimization_type, 'parabolic')) && ...
             epochs_in_band > 500;

if ~use_stream
    [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA] = ...
        local_legacy_path(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, ...
            refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold, ...
            smoothing_window_seconds, percentile_threshold, rank_truncation, opts);
    if cast_back
        cleaned_data = double(cleaned_data);
        if ~isempty(artifacts_data), artifacts_data = double(artifacts_data); end
    end
    return
end

%% ---------------- streaming path ----------------------------------------
pnts_original = size(eeg_data, 2);
epoch_samples = round(srate * epoch_size);

remainder = rem(pnts_original, epoch_samples);
if remainder ~= 0
    samples_to_pad = epoch_samples - remainder;
    eeg_data = [eeg_data, local_reflect_pad(eeg_data, samples_to_pad)];
end

[cleaned_padded, SENSAI_score, artifact_threshold_out, ~] = gedai_band_stream( ...
    eeg_data, srate, epoch_size, refCOV, artifact_threshold_type, optimization_type, ...
    signal_type, minThreshold, maxThreshold, percentile_threshold, rank_truncation, opts);

cleaned_data = cleaned_padded(:, 1:pnts_original);
clear cleaned_padded

% ENOVA per epoch, accumulated without ever building the artifact array.
% cleaned + artifacts == input by construction, so the removed signal is just
% the difference and never needs to be stored in full.
% Vectorised over epochs, in chunks. Two var() calls per epoch showed up as
% 364k calls / ~600 s in a whole-pipeline profile; one call per chunk instead.
% Chunked rather than whole-array so the difference never costs a full copy.
num_epochs_possible = floor(pnts_original / epoch_samples);
if num_epochs_possible > 0
    enova_per_epoch = zeros(1, num_epochs_possible);
    ep_per_chunk = max(1, floor(64*2^20 / max(N_EEG_electrodes * epoch_samples * 8, 1)));
    for cs = 1:ep_per_chunk:num_epochs_possible
        ce = min(cs + ep_per_chunk - 1, num_epochs_possible);
        sl = (cs-1)*epoch_samples + 1 : ce*epoch_samples;
        nc = ce - cs + 1;
        O  = reshape(eeg_data(:, sl), [], nc);
        A  = O - reshape(cleaned_data(:, sl), [], nc);
        vo = var(O, 0, 1); va = var(A, 0, 1);
        e  = zeros(1, nc); m = vo > 0; e(m) = va(m) ./ vo(m);
        enova_per_epoch(cs:ce) = e;
    end
    ENOVA = mean(enova_per_epoch);
else
    ENOVA = 0;
end

if opts.want_artifacts
    artifacts_data = eeg_data(:, 1:pnts_original) - cleaned_data;
else
    artifacts_data = [];
end

if cast_back
    cleaned_data = double(cleaned_data);
    if ~isempty(artifacts_data), artifacts_data = double(artifacts_data); end
end
end


% =====================================================================
function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA] = ...
    local_legacy_path(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, ...
        refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold, ...
        smoothing_window_seconds, percentile_threshold, rank_truncation, opts) %#ok<INUSL>
%LOCAL_LEGACY_PATH  Pre-streaming implementation, unchanged apart from the
%   optional threshold override.

N_EEG_electrodes = size(eeg_data, 1);

%% Pad and Epoch Data
pnts_original = size(eeg_data, 2);
epoch_samples = round(srate * epoch_size);

remainder = rem(pnts_original, epoch_samples);
if remainder ~= 0
    samples_to_pad = epoch_samples - remainder;
    padding = local_reflect_pad(eeg_data, samples_to_pad);
    eeg_data = [eeg_data, padding];
end

EEGdata_epoched = reshape(eeg_data, N_EEG_electrodes, epoch_samples, []);

shifting = epoch_samples / 2;
eeg_data_2 = eeg_data(:, (shifting+1):(end-shifting));
EEGdata_epoched_2 = reshape(eeg_data_2, N_EEG_electrodes, epoch_samples, []);
[~,~,N_epochs] = size(EEGdata_epoched);

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

[Evec,   Evald]   = local_gevd(EEGdata_epoched,   R_chol, gevd_rank, N_EEG_electrodes, epoch_samples);
[Evec_2, Evald_2] = local_gevd(EEGdata_epoched_2, R_chol, gevd_rank, N_EEG_electrodes, epoch_samples);

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

if ~isempty(opts.artifact_threshold_override)
    optimal_threshold_per_window(:) = opts.artifact_threshold_override;
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
cosine_weights = create_cosine_weights(N_EEG_electrodes, srate, epoch_size, 1);

artifact_threshold_2 = (artifact_threshold(1:end-1) + artifact_threshold(2:end)) / 2;
if isempty(artifact_threshold_2)
    artifact_threshold_2 = artifact_threshold;
end

[cleaned_data_1, artifacts_data_1, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold, refCOV, Evald, Evec, cosine_weights, signal_type, refCOV_reg, percentile_threshold);
[cleaned_data_2, artifacts_data_2, ~] = clean_EEG(EEGdata_epoched_2, srate, epoch_size, artifact_threshold_2, refCOV, Evald_2, Evec_2, cosine_weights, signal_type, refCOV_reg, percentile_threshold);

clear EEGdata_epoched_2 Evec_2 Evald_2

size_reconstructed_2 = size(cleaned_data_2, 2);
sample_end = size_reconstructed_2 - shifting;
cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
cleaned_data_2(:, sample_end+1:end) = cleaned_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);
artifacts_data_2(:, 1:shifting) = artifacts_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
artifacts_data_2(:, sample_end+1:end) = artifacts_data_2(:, sample_end+1:end) .* cosine_weights(:, (shifting+1):end);

cleaned_data = cleaned_data_1;
clear cleaned_data_1

artifacts_data = artifacts_data_1;
clear artifacts_data_1

cleaned_data(:, shifting+1:shifting+size_reconstructed_2) = cleaned_data(:, shifting+1:shifting+size_reconstructed_2) + cleaned_data_2;
clear cleaned_data_2

artifacts_data(:, shifting+1:shifting+size_reconstructed_2) = artifacts_data(:, shifting+1:shifting+size_reconstructed_2) + artifacts_data_2;
clear artifacts_data_2

cleaned_data = cleaned_data(:, 1:pnts_original);
artifacts_data = artifacts_data(:, 1:pnts_original);

%% Calculate final SENSAI score
[~, ~, SENSAI_score] = SENSAI(mean(artifact_threshold_out), refCOV, Evald, Evec, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold);

original_data = cleaned_data + artifacts_data;
epoch_samples = round(srate * epoch_size);
num_epochs_possible = floor(size(original_data, 2) / epoch_samples);
len_to_use = num_epochs_possible * epoch_samples;

original_epoched = reshape(original_data(:, 1:len_to_use), size(original_data, 1), epoch_samples, []);
artifacts_epoched = reshape(artifacts_data(:, 1:len_to_use), size(artifacts_data, 1), epoch_samples, []);

num_epochs = size(original_epoched, 3);
% one var() per array instead of two per epoch (see the streaming path)
original_flat  = reshape(original_epoched,  [], num_epochs);
artifacts_flat = reshape(artifacts_epoched, [], num_epochs);
var_orig = var(original_flat, 0, 1);
var_art  = var(artifacts_flat, 0, 1);
enova_per_epoch = zeros(1, num_epochs);
valid = var_orig > 0;
enova_per_epoch(valid) = var_art(valid) ./ var_orig(valid);

if num_epochs > 0
    ENOVA = mean(enova_per_epoch);
else
    ENOVA = 0;
end
end


% =====================================================================
function padding = local_reflect_pad(eeg_data, samples_to_pad)
%LOCAL_REFLECT_PAD  Reflect-pad, tiling when the pad is longer than the data.
if samples_to_pad <= 0
    padding = zeros(size(eeg_data, 1), 0, 'like', eeg_data);
    return;
end
if size(eeg_data, 2) == 0
    error('Cannot pad empty EEG data.');
end
padding = zeros(size(eeg_data, 1), samples_to_pad, 'like', eeg_data);
filled = 0;
while filled < samples_to_pad
    reflected_chunk = fliplr(eeg_data);
    chunk_len = min(size(reflected_chunk, 2), samples_to_pad - filled);
    padding(:, filled + 1:filled + chunk_len) = reflected_chunk(:, 1:chunk_len);
    filled = filled + chunk_len;
end
end


% =====================================================================
function [Evec, Evald] = local_gevd(X_epoched, R_chol, r, N, T)
%LOCAL_GEVD  Per-epoch generalized eigendecomposition against refCOV_reg.
K = size(X_epoched, 3);
Evec  = zeros(N, r, K, 'like', X_epoched);
Evald = zeros(r, K, 'like', X_epoched);
if K == 0
    return
end

truncated = (r < N);
chunk_size = max(1, min(500, floor(2^23 / max(N * T, 1))));

for cs = 1:chunk_size:K
    ce = min(cs + chunk_size - 1, K);
    blk = X_epoched(:,:,cs:ce);
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
