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
%
%   The arithmetic lives in clean_EEG_thresholds (set-up) and clean_EEG_epoch
%   (one epoch), which the streaming band engine calls directly so that it
%   never needs the whole epoched band in memory.

if nargin < 10, refCOV_reg_in = []; end
if nargin < 11, percentile_threshold = []; end

num_chans  = size(Evec, 1);
num_epochs = size(Evec, 3);

[threshold_array, artifact_threshold_out, refCOV_reg, magnitudes] = clean_EEG_thresholds( ...
    Evald, num_chans, artifact_threshold_in, refCOV, signal_type, refCOV_reg_in, percentile_threshold);

%% Cleaning EEG by removing outlying GEVD components
epoch_samples = round(srate * epoch_size);
artifacts = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
cleaned_epoched_data = zeros(size(EEGdata_epoched), 'like', EEGdata_epoched);
if nargin < 8 || isempty(cosine_weights)
    cosine_weights = create_cosine_weights(num_chans, srate, epoch_size, 1);
end

for i = 1:num_epochs
    [cleaned_epoched_data(:,:,i), artifacts(:,:,i)] = clean_EEG_epoch(EEGdata_epoched(:,:,i), i, ...
        num_epochs, magnitudes, threshold_array, Evec, refCOV_reg, cosine_weights, epoch_samples);
end

% Reshape data back to continuous form and return outputs
cleaned_data = reshape(cleaned_epoched_data, num_chans, []);
artifacts_data = reshape(artifacts, num_chans, []);

end
