function [cleaned_epoch, artifact_epoch] = clean_EEG_epoch(X, i, num_epochs, magnitudes, threshold_array, Evec, refCOV_reg, cosine_weights, epoch_samples, Ulf, gate)
%CLEAN_EEG_EPOCH  Clean one epoch of one epoch grid, as clean_EEG does.
%
%   [cleaned_epoch, artifact_epoch] = clean_EEG_epoch(X, i, num_epochs, ...
%       magnitudes, threshold_array, Evec, refCOV_reg, cosine_weights, ...
%       epoch_samples, Ulf, gate)
%
%   X is epoch i (channels x epoch_samples) of a grid of num_epochs epochs;
%   magnitudes, threshold_array and refCOV_reg come from clean_EEG_thresholds,
%   Evec is the N x r x K eigenvector array of the whole grid. The cosine taper
%   depends on i only through whether it is the first or last epoch of the grid.
%   Both outputs carry the taper.
%
%   Ulf, gate - optional plausibility floor, mirroring gedai_band_engine's
%   local_clean_block/local_npc90. Ulf is the eigenbasis of the raw leadfield
%   gram, sorted descending; gate is a count of leadfield principal components.
%   A component whose topography reaches 90% of its energy within that many
%   leadfield directions is a dipolar source and is kept, whatever the
%   threshold says. gate = 0 or [] (default) disables the floor and reproduces
%   the previous behaviour exactly.

if nargin < 10, Ulf = []; end
if nargin < 11 || isempty(gate), gate = 0; end

num_chans = size(Evec, 1);

% Only the suprathreshold components survive the masking that the
% original formulation applied to a full N x N spatial filter, and there
% are typically a handful of them. Indexing them directly is the same
% arithmetic on a much smaller matrix.
bad_indices = magnitudes(:,i) >= threshold_array(i);

if any(bad_indices)
    Evec_bad = Evec(:, bad_indices, i);
    if gate > 0
        %%% Plausibility floor - see gedai_band_engine/local_npc90 for the
        %%% rationale and the measured separation. Applied at cleaning time
        %%% only, so the threshold SENSAI optimised is untouched; the gate can
        %%% only remove less, never more.
        Evec_bad = Evec_bad(:, local_npc90(refCOV_reg, Evec_bad, Ulf) > gate);
    end
    num_bad = size(Evec_bad, 2);
else
    num_bad = 0;
end

if num_bad > 0
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
%   Duplicated from gedai_band_engine/local_npc90 because the two files are separate
%   entry points; keep them in step.
A = B * V;
nrm = sqrt(sum(A.^2, 1)); nrm(nrm == 0) = 1;
A = A ./ nrm;
E = (Ulf' * A).^2;
C = cumsum(E, 1) ./ max(sum(E, 1), realmin);
npc = sum(C < 0.90, 1) + 1;
end
