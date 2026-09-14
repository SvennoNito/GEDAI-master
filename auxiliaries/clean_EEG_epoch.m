function [cleaned_epoch, artifact_epoch] = clean_EEG_epoch(X, i, num_epochs, magnitudes, threshold_array, Evec, refCOV_reg, cosine_weights, epoch_samples)
%CLEAN_EEG_EPOCH  Clean one epoch of one epoch grid, as clean_EEG does.
%
%   [cleaned_epoch, artifact_epoch] = clean_EEG_epoch(X, i, num_epochs, ...
%       magnitudes, threshold_array, Evec, refCOV_reg, cosine_weights, epoch_samples)
%
%   X is epoch i (channels x epoch_samples) of a grid of num_epochs epochs;
%   magnitudes, threshold_array and refCOV_reg come from clean_EEG_thresholds,
%   Evec is the N x r x K eigenvector array of the whole grid. The cosine taper
%   depends on i only through whether it is the first or last epoch of the grid.
%   Both outputs carry the taper.

num_chans = size(Evec, 1);

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

artifact_epoch = Signal_to_remove;
cleaned_epoch = X - Signal_to_remove;

% Apply cosine windowing to mitigate edge effects from epoching
half_epoch = epoch_samples/2;
if i == 1
    cleaned_epoch(:, half_epoch+1:end) = cleaned_epoch(:, half_epoch+1:end) .* cosine_weights(:, half_epoch+1:end);
    artifact_epoch(:, half_epoch+1:end) = artifact_epoch(:, half_epoch+1:end) .* cosine_weights(:, half_epoch+1:end);
elseif i == num_epochs
    cleaned_epoch(:, 1:half_epoch) = cleaned_epoch(:, 1:half_epoch) .* cosine_weights(:, 1:half_epoch);
    artifact_epoch(:, 1:half_epoch) = artifact_epoch(:, 1:half_epoch) .* cosine_weights(:, 1:half_epoch);
else
    cleaned_epoch = cleaned_epoch .* cosine_weights;
    artifact_epoch = artifact_epoch .* cosine_weights;
end
end
