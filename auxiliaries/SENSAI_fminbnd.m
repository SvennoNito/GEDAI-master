function [optimalThreshold, maxSENSAIScore] = SENSAI_fminbnd(minThreshold, maxThreshold, refCOV, Evald, Evec, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold)

max_number_of_epochs = 500; % if EEG recording is long (default = 500 epochs)
number_of_epochs = size(Evec, 3);

if number_of_epochs > max_number_of_epochs
    rng(2,"twister") ; % for reproducibility
    random_epochs = randperm(number_of_epochs, max_number_of_epochs);
    Evald = Evald(:, random_epochs);
    Evec  = Evec(:, :, random_epochs);
end

if nargin < 10, percentile_threshold = []; end
sensaifunc = @(artifactThreshold) SENSAIObjective(artifactThreshold, refCOV, Evald, Evec, noise_multiplier, evecs_Template_cov, signal_type, SSI_top_PCs, percentile_threshold);
[optimalThreshold, negMaxSENSAIScore] = local_fminbnd(sensaifunc, minThreshold, maxThreshold, 1e-2);

maxSENSAIScore = -negMaxSENSAIScore;

    function objective = SENSAIObjective(artifact_threshold, refCOV, Evald, Evec, noise_multiplier_obj, evecs_Template_cov_obj, signal_type, SSI_top_PCs, percentile_threshold)
        % Compute the negative SENSAI score for the objective function
        [~, ~, SENSAI_score] = SENSAI(artifact_threshold, refCOV, Evald, Evec, noise_multiplier_obj, evecs_Template_cov_obj, signal_type, SSI_top_PCs, percentile_threshold);
        objective = -SENSAI_score;
    end
end
