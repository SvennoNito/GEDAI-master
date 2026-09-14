function [threshold_array, artifact_threshold_out, refCOV_reg, magnitudes] = clean_EEG_thresholds(Evald, num_chans, artifact_threshold_in, refCOV, signal_type, refCOV_reg_in, percentile_threshold)
%CLEAN_EEG_THRESHOLDS  Per-epoch eigenvalue cut-offs used by clean_EEG.
%
%   [threshold_array, artifact_threshold_out, refCOV_reg, magnitudes] = ...
%       clean_EEG_thresholds(Evald, num_chans, artifact_threshold_in, refCOV, ...
%                            signal_type, refCOV_reg_in, percentile_threshold)
%
%   The set-up half of clean_EEG, split out so that the streaming band engine
%   (gedai_band_engine) cleans epochs against exactly the same thresholds
%   without holding the whole band in memory. The per-epoch half is
%   clean_EEG_epoch. Evald is r x K, one eigenvalue column per epoch; pass
%   refCOV_reg_in or percentile_threshold as [] for the defaults.

num_epochs = size(Evald, 2);
magnitudes = abs(Evald);

%% Artifacting multiplication factor T1
correction_factor = 1.00;

if isscalar(artifact_threshold_in)
    artifact_threshold_in = repmat(artifact_threshold_in, 1, num_epochs);
end

T1_array = correction_factor * (105 - artifact_threshold_in) / 100;

%% Defining artifact threshold
if isempty(percentile_threshold)
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
if isempty(refCOV_reg_in)
    refCOV_local = real(refCOV);
    refCOV_local = (refCOV_local + refCOV_local') / 2;
    reg_lambda = 0.05;
    reg_val_local = trace(refCOV_local) / num_chans;
    refCOV_reg = (1-reg_lambda)*refCOV_local + reg_lambda*reg_val_local*eye(num_chans, 'like', refCOV_local);
    refCOV_reg = (refCOV_reg + refCOV_reg') / 2;
else
    refCOV_reg = refCOV_reg_in;
end

artifact_threshold_out = artifact_threshold_in;
end
