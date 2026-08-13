function [cov_signal_epoched, cov_noise_epoched, artifact_threshold_out, Treshold1] = clean_SENSAI(artifact_threshold_in, refCOV, Evald, Evec, signal_type, percentile_threshold)
%   This GEDAI function estimates signal and noise covariances analytically
%
%   NOTE: SENSAI no longer calls this. It computes the same subspace
%   similarities without ever materializing cov_signal/cov_noise, which are
%   two N x N x K arrays. This function is retained for diagnostics and for
%   callers that genuinely want the covariances themselves.
%
%   Evald is r x K (one eigenvalue column per epoch), Evec is N x r x K.
%
%%   Creative Commons License
%
% Copyright:  Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its CONTRIBUTORS
% may be used to endorse or promote products derived from this software without
% specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
% THE POSSIBILITY OF SUCH DAMAGE.

% --- PRE-ALLOCATION ---
num_chans  = size(Evec, 1);
num_epochs = size(Evec, 3);
all_evals_mat = abs(Evald);

%% Artifacting multiplication factor T1
correction_factor = 1.00;
T1 = correction_factor * (105 - artifact_threshold_in) / 100;

%% Defining artifact threshold
if nargin < 6 || isempty(percentile_threshold)
    if strcmpi(signal_type, 'eeg')
        percentile_threshold = 98;
    elseif strcmpi(signal_type, 'meg')
        percentile_threshold = 99;
    end
end
[threshold_val, Treshold1] = gedai_eig_threshold(Evald, num_chans, percentile_threshold, T1);

%% Compute Regularized Reference Covariance
% Replicate logic from GEDAI_per_band.m to ensure we have the correct B for B-orthogonality
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;
regularization_lambda = 0.05;
reg_val = trace(refCOV) / num_chans;
refCOV_reg = (1-regularization_lambda)*refCOV + regularization_lambda*reg_val*eye(num_chans, 'like', refCOV);
refCOV_reg = (refCOV_reg + refCOV_reg') / 2;

%% Cleaning EEG by removing outlying GEVD components
cov_signal_epoched = zeros(num_chans, num_chans, num_epochs, 'like', Evald);
cov_noise_epoched  = zeros(num_chans, num_chans, num_epochs, 'like', Evald);

for i = 1:num_epochs
    current_evals = all_evals_mat(:, i);

    % 'bad_indices' are indices of ARTIFACT components (Large Eigenvalues)
    bad_indices = current_evals >= threshold_val;

    if any(bad_indices)
        % V_inv = Evec' * refCOV_reg (from GEVD B-orthogonality), so the
        % rows of V_inv for the bad components are Evec(:,bad)' * refCOV_reg.
        Evec_bad = Evec(:, bad_indices, i);
        V_bad_rows = Evec_bad' * refCOV_reg;
        d_bad = current_evals(bad_indices);
        cov_noise_epoched(:,:,i) = V_bad_rows' * (V_bad_rows .* d_bad);
    end

    good_indices = ~bad_indices;
    if any(good_indices)
        Evec_good = Evec(:, good_indices, i);
        V_good_rows = Evec_good' * refCOV_reg;
        d_good = current_evals(good_indices);
        cov_signal_epoched(:,:,i) = V_good_rows' * (V_good_rows .* d_good);
    end

    % Enforce symmetry to allow fast symmetric eig solver downstream
    cov_noise_epoched(:,:,i)  = (cov_noise_epoched(:,:,i) + cov_noise_epoched(:,:,i)') / 2;
    cov_signal_epoched(:,:,i) = (cov_signal_epoched(:,:,i) + cov_signal_epoched(:,:,i)') / 2;
end

artifact_threshold_out = artifact_threshold_in;

end
