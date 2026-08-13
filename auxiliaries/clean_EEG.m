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

function [cleaned_data, artifacts_data, artifact_threshold_out] = clean_EEG(EEGdata_epoched, srate, epoch_size, artifact_threshold_in, refCOV, Evald, Evec, cosine_weights, signal_type, refCOV_reg_in, percentile_threshold)
%   This GEDAI function reconstructs the signal after removing artifactual components
%
%   Evald is r x K (one eigenvalue column per epoch), Evec is N x r x K.

% --- PRE-ALLOCATION ---
num_chans  = size(Evec, 1);
num_epochs = size(Evec, 3);
magnitudes = abs(Evald);

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

% One percentile over ALL eigenvalues (all channels x all epochs), scaled
% per epoch by T1_array, so that this matches exactly how SENSAI evaluated
% the threshold it optimized.
threshold_array = gedai_eig_threshold(Evald, num_chans, percentile_threshold, T1_array);

%% Compute refCOV_reg for B-orthogonal reconstruction
% V^{-T} = refCOV_reg * V  (from GEVD B-orthogonality: V'*B*V = I)
% This replaces the per-epoch O(n^3) backslash with matrix multiplies.
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

%% Cleaning EEG by removing outlying GEVD components
epoch_samples = round(srate * epoch_size);
artifacts = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
cleaned_epoched_data = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
if nargin < 8 || isempty(cosine_weights)
    cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
end
half_epoch = epoch_samples/2;

for i = 1:num_epochs
    X = EEGdata_epoched(:,:,i);

    % Only the suprathreshold components survive the masking that the
    % original formulation applied to a full N x N spatial filter, and there
    % are typically a handful of them. Indexing them directly is the same
    % arithmetic on a much smaller matrix.
    bad_indices = magnitudes(:,i) >= threshold_array(i);
    num_bad = sum(bad_indices);

    if num_bad > 0
        Evec_bad = Evec(:, bad_indices, i);
        if num_bad <= epoch_samples
            % Fold refCOV_reg into the (N x num_bad) basis first.
            Signal_to_remove = (refCOV_reg * Evec_bad) * (Evec_bad' * X);
        else
            % Cheaper to defer refCOV_reg to the (N x epoch_samples) result.
            Signal_to_remove = refCOV_reg * (Evec_bad * (Evec_bad' * X));
        end
    else
        Signal_to_remove = zeros(num_chans, size(X, 2), 'like', X);
    end

    artifacts(:,:,i) = Signal_to_remove;
    cleaned_epoch = X - Signal_to_remove;

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

% Reshape data back to continuous form and return outputs
cleaned_data = reshape(cleaned_epoched_data, num_chans, []);
artifacts_data = reshape(artifacts, num_chans, []);
artifact_threshold_out = artifact_threshold_in;

end
