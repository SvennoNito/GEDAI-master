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

function [cleaned_data, artifacts_data, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold_in, refCOV, Eval, Evec, cosine_weights, signal_type, refCOV_reg_in, percentile_threshold)
%   This GEDAI function reconstructs the signal after removing artifactual components

% --- PRE-ALLOCATION ---
num_chans = size(Eval, 1);
num_epochs = size(Eval, 3);
% Pre-allocate the array 
all_diagonals = zeros(num_chans * num_epochs, 1, 'like', Eval);
for i = 1:num_epochs
    start_idx = (i-1) * num_chans + 1;
    end_idx = i * num_chans;
    all_diagonals(start_idx:end_idx) = diag(Eval(:,:,i));
end
% Use the magnitude (a real value) for all subsequent calculations.
magnitudes = abs(all_diagonals);
log_Eig_val_all = log(magnitudes(magnitudes > 0)) + 100;


%% Artifacting multiplication factor T1
correction_factor = 1.00;

if isscalar(artifact_threshold_in)
    artifact_threshold_in = repmat(artifact_threshold_in, 1, num_epochs);
end

T1_array = correction_factor * (105 - artifact_threshold_in) / 100;

%% Defining artifact threshold
if nargin < 11 || isempty(percentile_threshold)
    if strcmpi(signal_type, 'eeg')
        percentile_threshold = 98;
    elseif strcmpi(signal_type, 'meg')
        percentile_threshold = 99;
    end
end

% Compute Treshold1 per epoch to match exactly how clean_SENSAI evaluates it:
% SENSAI uses the GLOBAL percentile of ALL eigenvalues in the window it was
% optimized on. We replicate that here by computing one percentile over ALL
% eigenvalues (all channels × all epochs) and scaling per-epoch by T1_array.
global_log_prctile = prctile(log_Eig_val_all, percentile_threshold);
Treshold1_array = T1_array * global_log_prctile;

% %%% Figure?
% figure('color', 'w')
% histogram(log_Eig_val_all)
% xline(prctile(log_Eig_val_all, 96), '-', '95%')
% xline(prctile(log_Eig_val_all, 98), '-', '98%')
% xline(prctile(log_Eig_val_all, 99), '-', '99%')
% xlabel('Eigenvalues (all chans x all epochs)')
% ylabel('#')

%% Compute refCOV_reg for B-orthogonal reconstruction (Change A)
% V^{-T} = refCOV_reg * V  (from GEVD B-orthogonality: V'*B*V = I)
% This replaces the per-epoch O(n^3) backslash with two O(n^2*T) matrix multiplies.
if nargin < 10 || isempty(refCOV_reg_in)
    refCOV_local = real(refCOV);
    refCOV_local = (refCOV_local + refCOV_local') / 2;
    reg_lambda = 0.05;
    reg_val_local = trace(refCOV_local) / num_chans;
    refCOV_reg = (1-reg_lambda)*refCOV_local + reg_lambda*reg_val_local*eye(num_chans, 'like', refCOV_local);
    refCOV_reg = (refCOV_reg + refCOV_reg') / 2;
else
    refCOV_reg = refCOV_reg_in;
end

%% Cleaning EEG by removing outlying GEVD components (Change A: mini-batched pagemtimes)
epoch_samples = round(srate * epoch_size);
artifacts = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
cleaned_epoched_data = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
if nargin < 8 || isempty(cosine_weights)
    cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
end
half_epoch = epoch_samples/2;

eeg_chunk_size = 500;
for chunk_start = 1:eeg_chunk_size:num_epochs
    chunk_end = min(chunk_start + eeg_chunk_size - 1, num_epochs);

    % Batch project all epochs in chunk: Evec' * data (N x T x K)
    proj_chunk = pagemtimes(Evec(:,:,chunk_start:chunk_end), 'transpose', ...
                             EEGdata_epoched(:,:,chunk_start:chunk_end), 'none');

    for k = 1:(chunk_end - chunk_start + 1)
        i = chunk_start + k - 1;

        % Keep only artifact-component activations (zero out signal rows)
        signal_indices = abs(diag(Eval(:,:,i))) < exp(Treshold1_array(i) - 100);
        proj = proj_chunk(:,:,k);
        proj(signal_indices, :) = 0;

        % B-orthogonal solve: V^{-T} * proj = refCOV_reg * V * proj
        Signal_to_remove = refCOV_reg * (Evec(:,:,i) * proj);

        artifacts(:,:,i) = Signal_to_remove;
        cleaned_epoch = EEGdata_epoched(:,:,i) - Signal_to_remove;

        % Apply cosine windowing to mitigate edge effects from epoching
        if i == 1
            cleaned_epoch(:, half_epoch+1:end) = cleaned_epoch(:, half_epoch+1:end) .* cosine_weights(:, half_epoch+1:end);
            artifacts(:, half_epoch+1:end, i) = artifacts(:, half_epoch+1:end, i) .* cosine_weights(:, half_epoch+1:end);
        elseif i == num_epochs
            cleaned_epoch(:, 1:half_epoch) = cleaned_epoch(:, 1:half_epoch) .* cosine_weights(:, 1:half_epoch);
            artifacts(:, 1:half_epoch, i) = artifacts(:, 1:half_epoch, i) .* cosine_weights(:, 1:half_epoch);
        else
            cleaned_epoch = cleaned_epoch .* cosine_weights;
            artifacts(:,:,i) = artifacts(:,:,i) .* cosine_weights;
        end

        cleaned_epoched_data(:,:,i) = cleaned_epoch;
    end
end

% Reshape data back to continuous form and return outputs
cleaned_data = reshape(cleaned_epoched_data, num_chans, []);
artifacts_data = reshape(artifacts, num_chans, []);
artifact_threshold_out = artifact_threshold_in;

end