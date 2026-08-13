function [threshold_val, Treshold1, pct_effective] = gedai_eig_threshold(evals, num_chans, percentile_threshold, T1)
%GEDAI_EIG_THRESHOLD  Artifact cut-off on the generalized eigenvalue spectrum.
%
%   Single definition of the eigenvalue threshold, shared by clean_EEG,
%   SENSAI and clean_SENSAI. Those three must agree exactly: SENSAI optimizes
%   a threshold that clean_EEG then applies, so any drift between them
%   silently decouples the chosen optimum from what is actually removed.
%
%   Inputs
%     evals                - r x K matrix of generalized eigenvalues, one
%                            column per epoch (magnitude is taken here).
%     num_chans            - full channel count N. When r < N the epoch
%                            covariance was rank deficient and the GEVD was
%                            truncated to its r supported components; the
%                            remaining (N - r) per epoch are null space.
%     percentile_threshold - percentile of the eigenvalue pool, e.g. 98.
%     T1                   - scalar or 1 x K multiplier, (105 - threshold)/100.
%
%   Outputs
%     threshold_val        - eigenvalue cut-off, same shape as T1.
%     Treshold1            - the cut-off in the shifted log domain.
%     pct_effective        - percentile actually applied to the pool passed in.
%
%   The percentile is defined over the FULL N x K eigenvalue pool, which
%   includes the null-space eigenvalues that a truncated GEVD never forms.
%   Taking the same percentile of the truncated pool would land on a
%   different order statistic and move the threshold, so the percentile is
%   remapped onto the equivalent one. With r == N the remap is the identity,
%   which reproduces the original behaviour exactly.

mag = abs(evals);
log_vals = log(mag(mag > 0)) + 100;

if isempty(log_vals)
    Treshold1     = zeros(size(T1), 'like', T1);
    threshold_val = zeros(size(T1), 'like', T1);
    pct_effective = percentile_threshold;
    return
end

n_kept = numel(log_vals);
n_null = max(0, num_chans - size(evals, 1)) * size(evals, 2);

if n_null > 0
    n_effective   = n_kept + n_null;
    pct_effective = (percentile_threshold/100 * n_effective - n_null) / n_kept * 100;
    % Only meaningful while the percentile lands above the null space, i.e.
    % percentile_threshold/100 > (N - r)/N. Clamping keeps a pathological
    % configuration (very low percentile on a very rank-deficient band)
    % well defined rather than producing a complex or negative index.
    pct_effective = min(100, max(0, pct_effective));
else
    pct_effective = percentile_threshold;
end

Treshold1     = T1 .* prctile(log_vals, pct_effective);
threshold_val = exp(Treshold1 - 100);
end
