%   GEDAI() - This is the main function used to denoise EEG da12ta using
%   generalized eigenvalue decomposition coupled with an EEG leadfield matrix
%
%   GEDAI (Generalized Eigenvalue Deartifacting Instrument)
%
% Usage:
% Example 1: Using all default values (and no bad epoch rejection)
%    >>  [EEG] = GEDAI(EEG);
%
% Example 2: Defining some parameters
%    >>  [EEG] = GEDAI(EEG, 'auto', 12, 0.5, 'precomputed', true, false, 0.9);
%
% Example 3: Using a "custom" [channel x channel] reference matrix 
%    >>  [EEG] = GEDAI(EEG, 'auto', 12, 0.5, your_refCOV);
%
% Inputs: 
% 
%   EEGin                       - EEG data in EEGlab format
% 
%   artifact_threshold_type     - Variable determining deartifacting
%                                 strength. Stronger threshold type
%                                 ("auto+") might remove more noise at the
%                                 expense of signal, while milder threshold
%                                 ("auto-") might retain more signal at the 
%                                 expense of noise. Possible levels:
%                                 "auto-", "auto" or "auto+". 
%                                 Default is "auto".
%                             
%   epoch_size_in_cycles        - Epoch size in number of wave cycles for each
%                                 wavelet band. Default is 12.
%
%   lowcut_frequency            - Low-cut frequency in Hz. Wavelet bands below this
%                                 frequency will be excluded. Default is 0.5 Hz.
% 
%   ref_matrix_type             - Matrix used as a reference for deartifacting.
%
%                                  The default "precomputed" uses a BEM leadfield for
%                                  standard electrode locations precomputed through 
%                                  OPENMEEG (343 electrodes) based on 10-5 system. 
%
%                                 "interpolated" uses the precomputed leadfield and 
%                                 interpolates it to non-standard electrode locations.
%
%                                 "warped" uses an EEGLAB/Fieldtrip BEM surface source model 
%                                 (Colin27) warped to the current electrode locations.
%
%                                 Altenatively, you can input a "custom" covariance matrix
%                                 (with dimensions channel x channel) via a matlab variable
% 
% 
%   parallel                    - Boolean for using parallel ('multicore') processing 
% 
%   visualize_artifacts         - Boolean for artifact visualization 
%                                 using vis_artifacts function from the ASR toolbox
%
%   ENOVA_threshold_per_epoch   - Threshold for rejecting epochs based on Explained
%                                 Noise Variance (ENOVA). Epochs with ENOVA >
%                                 ENOVA_threshold_per_epoch will be removed. Default is inf
%                                 (no rejection).
%
%   ENOVA_threshold_per_channel - Threshold for rejecting channels based on Explained
%                                 Noise Variance (ENOVA). Channels with ENOVA >
%                                 ENOVA_threshold_per_channel will be removed. Default is inf
%                                 (no rejection).
%
%   signal_type                 - Type of signal: 'eeg' or 'meg'. Default is 'eeg'.
%                                 For EEG, average referencing is applied.
%                                 For MEG, average referencing is skipped.
%
%   smoothing_window_seconds    - Window size (in seconds) for sliding threshold adaptation 
%                                 to account for signal non-stationarities over time. 
%                                 Set to Inf (default) to use a fixed global threshold.
%
%   broadband_only              - Boolean. If true, only the broadband denoising pass is
%                                 run, skipping wavelet band decomposition. Faster but
%                                 less thorough. Default is false.
%
%   percentile_threshold        - Percentile (0-100) of the eigenvalue distribution used
%                                 as the artifact detection cutoff in GEDAI_per_band,
%                                 clean_EEG, and clean_SENSAI. Lower values = more
%                                 aggressive cleaning (more components removed); higher
%                                 values = more conservative cleaning. Default is 98 for
%                                 EEG and 99 for MEG.
%
%   broadband_minThreshold      - Minimum threshold value for broadband denoising pass.
%                                 Lower values allow more aggressive broadband cleaning.
%                                 Default is -2.
%
%   compute_SENSAI              - Boolean. If true (default), runs SENSAI_basic after
%                                 cleaning to compute SENSAI_score, mean_ENOVA, and
%                                 ENOVA_per_epoch. Set to false to skip this step and
%                                 return NaN for those outputs (useful when scoring will
%                                 be done externally).
%
% Outputs:
% 
%   EEGclean                - Cleaned EEG data in EEGLab struct format
% 
%   EEGartifacts            - EEG data containing only the removed artifacts
%                             (i.e. noise that was removed from EEGin)
%                             EEGin.data = EEGclean.data + EEGartifacts.data
% 
%   SENSAI_score            - Relative denoising quality score (%)
%
%   SENSAI_score_per_band   - Relative denoising quality score per band (%)
% 
%   artifact_threshold_per_band  - Vector of artifact thresholds used for each 
%                                  frequency band, starting with the broadband
%                                  approx: [broadband gamma beta alpha theta delta etc.]
%
%   mean_ENOVA              - Mean Explained Noise Variance (ENOVA) across all epochs.
%                             ENOVA is the variance of the removed noise, expressed as a 
%                             proportion of the variance of the original EEG data.
%
%   ENOVA_per_epoch         - Vector of ENOVA values for each epoch.
% 
%   com                     - output logging to EEG.history

% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI) v 1.6]
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

function [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band, ENOVA_per_channel, bandRemovedField]=GEDAI(EEGin, artifact_threshold_type, broadband_artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, parallel, visualize_artifacts, ENOVA_threshold_per_epoch, ENOVA_threshold_per_channel, signal_type, smoothing_window_seconds, broadband_epoch_size, broadband_only, percentile_threshold, broadband_minThreshold, compute_SENSAI, varargin)

if nargin < 2 || isempty(artifact_threshold_type)
    artifact_threshold_type = 'auto';
end
if nargin < 3 || isempty(broadband_artifact_threshold_type)
    broadband_artifact_threshold_type = 'auto-';
end
if nargin < 4 || isempty(epoch_size_in_cycles)
    epoch_size_in_cycles = 12;  % Note: Number of wave CYCLES per epoch across wavelet bands (default = 12 cycles)
end
if nargin < 5 || isempty(lowcut_frequency)
    lowcut_frequency = 0.5; %  exclude all wavelet bands below this frequency (default = 0.5 Hz)
end
if nargin < 6 || isempty(ref_matrix_type)
    ref_matrix_type = 'precomputed';
end
if nargin < 7 || isempty(parallel)
    parallel = true;
end
if nargin < 8 || isempty(visualize_artifacts)
    visualize_artifacts = false;
end
if nargin < 9 || isempty(ENOVA_threshold_per_epoch)
    ENOVA_threshold_per_epoch = inf; % If empty, set to infinity to disable rejection
end
if nargin < 10 || isempty(ENOVA_threshold_per_channel)
    ENOVA_threshold_per_channel = inf; % If empty, set to infinity to disable rejection
end

% Parse hidden internal arguments. varargin{1} is precomputed_ENOVA_per_epoch (Pass 2
% recursion). varargin{2} is a struct of caller overrides: it is merged into
% gedai_band_opts below, and may additionally carry
%   memory_budget_bytes  what the run may allocate on top of what it already holds.
%                        Default: what the machine reports free. Set it to model a
%                        smaller machine than the one running, so the low-memory path
%                        can be exercised and measured anywhere.
%   materialise_bands    override the budget decision for the band buffer.
%   keep_highpass        override the budget decision for the high-passed signal.
% Anything else in it is passed to the band engine (want_artifacts, precision,
% block_epochs, force_legacy, verbose, ...). Before this existed the struct was
% accepted and silently dropped.
precomputed_ENOVA_per_epoch = [];
caller_band_opts = struct();
if ~isempty(varargin)
    precomputed_ENOVA_per_epoch = varargin{1};
end
if numel(varargin) > 1 && ~isempty(varargin{2})
    caller_band_opts = varargin{2};
    if ~isstruct(caller_band_opts)
        error('GEDAI:bandOpts', 'The band-options argument must be a struct.');
    end
end
if nargin < 11 || isempty(signal_type)
    signal_type = 'eeg';
end
if nargin < 12 || isempty(smoothing_window_seconds)
    smoothing_window_seconds = Inf; % default: use whole file (no sliding window)
end
if nargin < 13 || isempty(broadband_epoch_size)
    broadband_epoch_size = 2; % default: use whole file (no sliding window)
end
if nargin < 14 || isempty(broadband_only)
    broadband_only = false;
end
if nargin < 15 || isempty(percentile_threshold)
    percentile_threshold = []; % default: auto-derived from signal_type in clean_EEG / clean_SENSAI
end
if nargin < 16 || isempty(broadband_minThreshold)
    broadband_minThreshold = -2;
end
if nargin < 17 || isempty(compute_SENSAI)
    compute_SENSAI = true;
end
% Validate signal_type
if ~ismember(lower(signal_type), {'eeg', 'meg'})
    error('signal_type must be either ''eeg'' or ''meg''');
end
signal_type = lower(signal_type);

%% Parallelisation axis.
%%   parallel = true      parfor over frequency bands (the original behaviour).
%%                        At most nBands tasks, badly unbalanced (the widest
%%                        band costs ~7x the narrowest), and every worker gets
%%                        a broadcast copy of the full-length band data.
%%   parallel = 'blocks'  serial band loop, parfor over time blocks inside each
%%                        band. Once the threshold is fixed the epochs are
%%                        independent, so a band splits into many equal tasks
%%                        and per-worker memory is one block rather than one
%%                        band. Nested parfor does not nest, so the band loop
%%                        has to give way.
use_block_parallel = (ischar(parallel) || isstring(parallel)) && strcmpi(char(parallel), 'blocks');
if use_block_parallel
    parallel = false;
end

%%% Per-band removed-field capture (multivariate-prep issue #25, ADR 0004): opt-in and
%%% windows-only. bandRemovedField defaults to {} so every return path - including the
%%% two-pass bad-channel mode's early return below, which this feature does not thread
%%% into - has it defined. capture_windows is [nWin x 2] (sampleStart, sampleEnd) in
%%% EEGin's own absolute column space; passing none (the default) means gedai_band_opts
%%% below carries capture_windows = zeros(0, 2), which costs gedai_band_engine nothing.
bandRemovedField = {};
capture_windows = zeros(0, 2);
if isfield(caller_band_opts, 'capture_windows') && ~isempty(caller_band_opts.capture_windows)
    capture_windows = caller_band_opts.capture_windows;
end
if ~isempty(capture_windows)
    if ENOVA_threshold_per_channel < inf
        error('GEDAI:captureWindowsUnsupported', ...
            ['CaptureWindows (multivariate-prep issue #25) is not supported together with the ' ...
             'two-pass bad-channel rejection mode (ENOVA_threshold_per_channel < Inf).']);
    end
    if parallel && ~use_block_parallel
        error('GEDAI:captureWindowsUnsupported', ...
            ['CaptureWindows (multivariate-prep issue #25) requires the serial wavelet-band loop ' ...
             '(parallel = ''blocks''); the band-parallel loop does not return per-band captures.']);
    end
end

p = fileparts(which('GEDAI'));
addpath(fullfile(p, 'auxiliaries'));
tStart = tic;

% =========================================================================
% TWO-PASS APPROACH FOR CHANNEL REJECTION
% =========================================================================
if ENOVA_threshold_per_channel < inf
    disp([newline '==================================================']);
    disp('GEDAI BAD-CHANNEL REJECTION MODE: Identifying and excluding bad channels');
    disp('==================================================');
    
    % --- PRE-PASS: Flat Channel Identification ---
    disp([newline '--- PRE-PASS: Flat Channel Identification ---']);
    flat_tolerance = 1e-7;
    EEG_data_2D = reshape(EEGin.data, size(EEGin.data, 1), []);
    channel_diff_std = std(diff(EEG_data_2D, 1, 2), 0, 2);
    flat_channels = find(channel_diff_std < flat_tolerance);
    
    if ~isempty(flat_channels)
        disp(['Found ' num2str(length(flat_channels)) ' flat channel(s). They will be automatically excluded.']);
    end
    
    % --- PASS 1 ---
    disp([newline '--- PASS 1: Identifying noisy channels ---']);
    % Run GEDAI on non-flat channels to identify noisy channels
    good_channels_p1 = setdiff(1:size(EEGin.data, 1), flat_channels);
    
    EEG_p1 = EEGin;
    if ~isempty(flat_channels)
        EEG_p1.data(flat_channels, :, :) = [];
        EEG_p1.chanlocs(flat_channels) = [];
        EEG_p1.nbchan = size(EEG_p1.data, 1);
        
        ref_matrix_type_p1 = ref_matrix_type;
        if ~ischar(ref_matrix_type_p1)
            ref_matrix_type_p1(flat_channels, :) = [];
            ref_matrix_type_p1(:, flat_channels) = [];
        end
    else
        ref_matrix_type_p1 = ref_matrix_type;
    end
    
    % Run GEDAI with channel rejection disabled (inf) to identify bad channels
    % Also disable epoch rejection in pass 1 so channel variance isn't computed on incomplete data
    [~, ~, ~, ~, ~, mean_ENOVA_p1, ENOVA_per_epoch_p1, ~, ~, ENOVA_per_channel_val_p1] = GEDAI(EEG_p1, artifact_threshold_type, broadband_artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type_p1, parallel, false, inf, inf, signal_type, smoothing_window_seconds);
%     [cleaned_broadband_data, ~, broadband_sensai, broadband_thresh, broadband_ENOVA] = GEDAI_per_band(double(EEGavRef.data), EEGavRef.srate, EEGavRef.chanlocs, broadband_artifact_threshold_type, broadband_epoch_size, refCOV, broadband_optimization_type, parallel, signal_type, broadband_minThreshold, broadband_maxThreshold, smoothing_window_seconds, percentile_threshold, [], gedai_band_opts);

    clear EEGclean_p1 EEGartifacts_p1; % Free memory
    
    noisy_channels_p1_idx = find(ENOVA_per_channel_val_p1 > ENOVA_threshold_per_channel);
    noisy_channels = good_channels_p1(noisy_channels_p1_idx);
    
    % Full list of channels to remove
    channels_to_remove = union(flat_channels(:), noisy_channels(:));
    
    % Construct the full ENOVA output array (flat channels get Inf)
    ENOVA_per_channel_val = nan(size(EEGin.data, 1), 1);
    ENOVA_per_channel_val(good_channels_p1) = ENOVA_per_channel_val_p1;
    ENOVA_per_channel_val(flat_channels) = Inf; % Flat channels are completely artificial
    
    if isempty(channels_to_remove)
        disp([newline 'No bad channels found. Proceeding with standard pass.']);
        % Set to inf to prevent recursion, and let the rest of the script run normally
        ENOVA_threshold_per_channel = inf;
    else
        disp([newline 'Found ' num2str(length(channels_to_remove)) ' bad channels. Removing them and running Pass 2...']);
        
        % Remove bad channels
        EEG_reduced = EEGin;
        EEG_reduced.data(channels_to_remove, :, :) = [];
        EEG_reduced.chanlocs(channels_to_remove) = [];
        EEG_reduced.nbchan = size(EEG_reduced.data, 1);
        
        % Update reference matrix if it's a custom matrix
        if ~ischar(ref_matrix_type)
            ref_matrix_type_reduced = ref_matrix_type;
            ref_matrix_type_reduced(channels_to_remove, :) = [];
            ref_matrix_type_reduced(:, channels_to_remove) = [];
        else
            ref_matrix_type_reduced = ref_matrix_type;
        end
        
        % --- PASS 2 ---
        disp([newline '--- PASS 2: Processing reduced data with global epoch thresholds ---']);
        [EEGclean, EEGartifacts, SENSAI_score, SENSAI_score_per_band, artifact_threshold_per_band, mean_ENOVA, ENOVA_per_epoch, com, ENOVA_per_band] = ...
            GEDAI(EEG_reduced, artifact_threshold_type, broadband_artifact_threshold_type, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type_reduced, parallel, false, ENOVA_threshold_per_epoch, inf, signal_type, smoothing_window_seconds, broadband_epoch_size, false, percentile_threshold, broadband_minThreshold, compute_SENSAI, ENOVA_per_epoch_p1);
        
        % --- INTERPOLATION ---
        disp([newline '--- INTERPOLATING BAD CHANNELS ---']);
        % Use EEGLAB's eeg_interp to interpolate missing channels back to the original montage
        EEGclean = eeg_interp(EEGclean, EEGin.chanlocs, 'spherical');
        
        % Reconstruct true EEGartifacts to perfectly preserve the fundamental invariant: original = clean + artifacts
        EEGartifacts = EEGclean;
        if isfield(EEGclean.etc, 'GEDAI') && isfield(EEGclean.etc.GEDAI, 'samples_to_keep')
            kept_samples = EEGclean.etc.GEDAI.samples_to_keep;
            original_data_kept = EEGin.data(:, kept_samples);
        else
            original_data_kept = EEGin.data;
        end
        EEGartifacts.data = original_data_kept - EEGclean.data;
        
        % Re-apply average reference after interpolation for EEG
        if strcmp(signal_type, 'eeg')
            disp('Re-applying average reference after interpolation...');
            EEGclean = GEDAI_nonRankDeficientAveRef(EEGclean);
            EEGartifacts = GEDAI_nonRankDeficientAveRef(EEGartifacts);
            % Update invariant reference for ENOVA calculation
            original_data_kept = EEGclean.data + EEGartifacts.data;
        end
        
        % Because epoch rejection permanently shortens the dataset, we cannot recalculate
        % the global mean ENOVA on the final output without mathematically losing the 
        % massive variance of the rejected bad epochs.
        % However, Pass 1 evaluated the full-topology global noise profile
        % perfectly across all channels and epochs BEFORE any data was discarded!
        mean_ENOVA = mean_ENOVA_p1;
        
        % Store the channel ENOVA and removed channels
        EEGclean.etc.GEDAI.ENOVA_per_channel = ENOVA_per_channel_val;
        EEGclean.etc.GEDAI.bad_channels_removed = channels_to_remove;
        EEGclean.etc.GEDAI.mean_ENOVA = mean_ENOVA;
        
        disp([newline '==================================================']);
        disp('GEDAI BAD-CHANNEL REJECTION MODE: Final Global Statistics');
        disp('==================================================');
        disp(['Global SENSAI Score: ' num2str(round(SENSAI_score, 1)) ' %']);
        disp(['Global Mean ENOVA: ' num2str(round(mean_ENOVA*100, 2, 'significant')) ' %']);
        disp(['Bad channels rejected: ' num2str(length(channels_to_remove)) ' (' num2str(round(length(channels_to_remove)/size(EEGin.data,1)*100, 1)) ' %)']);
        if isfield(EEGclean.etc, 'GEDAI') && isfield(EEGclean.etc.GEDAI, 'epochs_rejected')
            disp(['Bad epochs rejected: ' num2str(round(EEGclean.etc.GEDAI.percentage_rejected, 1)) ' % (' num2str(EEGclean.etc.GEDAI.epochs_rejected) ' out of ' num2str(EEGclean.etc.GEDAI.total_epochs) ' epochs)']);
        end
        
        ENOVA_per_channel = ENOVA_per_channel_val; % Provide output variable
        
        % Update com to reflect the original call
        if ~ischar(ref_matrix_type)
            ref_matrix_type_str = 'custom';
        else
            ref_matrix_type_str = ref_matrix_type;
        end
        com = sprintf('EEG = GEDAI(EEG, ''%s'', ''%s'', %s,  %s, ''%s'', %d,  %d, %s, %s, ''%s'');', ...
            artifact_threshold_type, broadband_artifact_threshold_type, num2str(epoch_size_in_cycles), num2str(lowcut_frequency), ref_matrix_type_str, parallel, visualize_artifacts, num2str(ENOVA_threshold_per_epoch), num2str(ENOVA_threshold_per_channel), signal_type);
        EEGclean = eegh(com, EEGclean);
        
        % --- FINAL VISUALIZATIONS ON FULL INTERPOLATED DATA ---
        if visualize_artifacts
            EEGclean_for_vis = EEGclean;
            if isfield(EEGclean.etc, 'GEDAI') && isfield(EEGclean.etc.GEDAI, 'samples_to_keep')
                EEGclean_for_vis.etc.clean_sample_mask = EEGclean.etc.GEDAI.samples_to_keep;
            end
            vis_artifacts(EEGclean_for_vis, EEGin, 'ScaleBy', 'noscale', 'YScaling', 3*mad(EEGin.data(:)));
            
            % Plot sliding thresholds if applicable
            if smoothing_window_seconds ~= Inf && isfield(EEGclean.etc.GEDAI, 'artifact_threshold_array_per_band')
                plot_title = ['GEDAI Sliding Thresholds (' artifact_threshold_type ' | Window: ' num2str(smoothing_window_seconds) ' s | SENSAI: ' num2str(round(SENSAI_score, 1)) '%)'];
                figure('Color', 'w', 'Name', plot_title);
                num_plots = length(EEGclean.etc.GEDAI.artifact_threshold_array_per_band);
                
                num_cols = min(num_plots, 3);
                num_rows = ceil(num_plots / num_cols);
                tiledlayout(num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
                sgtitle(plot_title, 'FontSize', 12, 'FontWeight', 'bold');
                
                band_colors = turbo(max(num_plots, 1));
                for i = 1:num_plots
                    nexttile;
                    thresh_array = EEGclean.etc.GEDAI.artifact_threshold_array_per_band{i};
                    
                    if i == 1
                        current_epoch_size = EEGclean.etc.GEDAI.broadband_epoch_size;
                    else
                        current_epoch_size = EEGclean.etc.GEDAI.epoch_sizes_per_wavelet_band(i-1);
                    end
                    
                    time_axis_minutes = (1:length(thresh_array)) * current_epoch_size / 60;
                    plot(time_axis_minutes, thresh_array, '-', 'Color', band_colors(i,:), 'LineWidth', 2);
                    
                    title(EEGclean.etc.GEDAI.freq_str_cell{i}, 'FontSize', 12);
                    
                    if i > num_plots - num_cols
                        xlabel('Time (Minutes)', 'FontSize', 10);
                    end
                    ylabel('Threshold', 'FontSize', 10);
                    grid on;
                    ylim([-1.9, 10]);
                end
            end
        end
        
        return; % End here for two-pass
    end
end
% =========================================================================

% Display signal type being processed
channel_type=EEGin.chanlocs(1).type;
if strcmp(signal_type, 'eeg')
    disp([newline 'GEDAI denoising of ' channel_type ' : '  num2str(size(EEGin.data,1)) ' channels']);
elseif strcmp(signal_type, 'meg')
    disp([newline 'GEDAI denoising of '  channel_type ' : ' num2str(size(EEGin.data,1)) ' channels']);
end  

% -- Handle Epoched Data --
is_epoched = false;
if EEGin.trials > 1 && ndims(EEGin.data) == 3
    is_epoched = true;
    disp('Epoched data detected. Converting to continuous for GEDAI processing...');
    original_EEG = EEGin; % Save to restore epoch structure later
    EEGin = eeg_epoch2continuous(EEGin);
end

% -- Ensure epoch size results in an even number of samples (for broadband)
%  broadband_epoch_size = 2; % Note: IN SECONDS (this is now only the DEFAULT for broadband)
if rem(broadband_epoch_size*EEGin.srate, 2) ~= 0
    ideal_total_samples_double = broadband_epoch_size * EEGin.srate;
    nearest_integer_samples = round(ideal_total_samples_double);
    if rem(nearest_integer_samples, 2) ~= 0
        if abs(ideal_total_samples_double - (nearest_integer_samples - 1)) < abs(ideal_total_samples_double - (nearest_integer_samples + 1))
            target_total_samples_int = nearest_integer_samples - 1;
        else
            target_total_samples_int = nearest_integer_samples + 1;
        end
    else
        target_total_samples_int = nearest_integer_samples;
    end
    broadband_epoch_size = target_total_samples_int / EEGin.srate;
end

%% Ensure double input (initially)
% The input stays in the precision it arrived in, and the double-precision,
% average-referenced, high-passed signal (EEGavRef.data) is built from it by
% local_prepare_avref. That signal is read by the broadband pass and, at the
% very end, for the artifact signal and ENOVA. Whether it is held across the
% wavelet bands or released and rebuilt afterwards is decided from the memory
% budget below: holding it costs one double copy of the recording through the
% stage where memory peaks, rebuilding it costs a second average reference and
% high-pass. Both give the same values, from the same input and the same code.
% Work arrays are processed in pieces of about piece_bytes, so that elementwise
% temporaries never approach the size of the recording.
%
% One group-sized double copy, the unit every decision below is measured in.
group_bytes = numel(EEGin.data) * 8;
memory_budget = local_memory_budget(caller_band_opts);
fprintf(['GEDAI memory: one copy of this recording is %.2f GB; budget for new ' ...
    'allocations is %.2f GB.\n'], group_bytes / 2^30, memory_budget / 2^30);

% Pieces stay small when the budget is tight and widen when it is not: the
% average reference works one column at a time and the high-pass one channel at
% a time, so the piece boundary never enters a value, only how many temporaries
% are alive. 256 MB was the fixed size before.
piece_bytes = 256 * 2^20;
if memory_budget > 8 * group_bytes
    piece_bytes = 2 * 2^30;
end
EEGavRef = EEGin;
EEGavRef.data = [];
apply_avg_ref = false;

%% Pre-processing
if strcmp(signal_type, 'eeg')
    % Check if data is already average referenced (Standard or via EEGLAB metadata)
    % (the sums are taken over double data, as before, a block of samples at a time)
    max_ref_offset = local_max_reference_offset(EEGin.data, piece_bytes);
    is_standard_avg_ref = max_ref_offset < 1e-5;

    if is_standard_avg_ref
        disp([newline 'Data is already average referenced. Skipping internal average referencing.']);

    elseif max_ref_offset < 1e-5
        % Corrected: Removed assignment and evaluated the math directly
        disp([newline 'Data matches non rank-deficient average reference definition. Skipping internal average referencing.']);

    else
        % non rank-deficient average referencing (GEDAI_nonRankDeficientAveRef),
        % applied to the data in local_prepare_avref
        apply_avg_ref = true;
        EEGavRef.ref = 'average';
        for chIdx = 1:EEGavRef.nbchan
            EEGavRef.chanlocs(chIdx).ref = 'average';
        end
    end

end


%% Create Reference Covariance Matrix (refCOV)

if ~ischar(ref_matrix_type)
    refCOV = ref_matrix_type; % Use custom covariance matrix
    disp([newline 'Using custom covariance matrix']);

 
else
    switch ref_matrix_type
        case 'precomputed'
        disp([newline 'GEDAI Leadfield model: BEM precomputed for EEG'])
            L=load('fsavLEADFIELD_4_GEDAI.mat');
            electrodes_labels = {EEGin.chanlocs.labels};
            template_electrode_labels = {L.leadfield4GEDAI.electrodes.Name};
            
            % Extract matching substrings from EEG labels
            chanidx = zeros(1, length(electrodes_labels));
            for i = 1:length(electrodes_labels)
                eeg_label = electrodes_labels{i};
                % Try direct match first
                [found, idx] = ismember(lower(eeg_label), lower(template_electrode_labels));
                if found
                    chanidx(i) = idx;
                else
                    % Search for template labels within the EEG label
                    for j = 1:length(template_electrode_labels)
                        template_label = template_electrode_labels{j};
                        % Case-insensitive substring search
                        if contains(lower(eeg_label), lower(template_label))
                            chanidx(i) = j;
                            break;
                        end
                    end
                end
            end
            
            if any(chanidx == 0)
                missing_labels = strjoin(electrodes_labels(chanidx == 0), ', ');
                error(['Electrode labels not found: ' missing_labels '. Either remove them using ''Edit ->Select data'' or select the ''interpolated'' leadfield matrix for non-standard locations.']);
            end
            refCOV = L.leadfield4GEDAI.gram_matrix_avref(chanidx,chanidx);

        case 'interpolated'
    % 1. Verification of Spatial Locations
    % We check if the number of populated X and sph_theta coordinates 
    % matches the actual number of channels.
    num_chans = length(EEGavRef.chanlocs);
    has_cartesian = length([EEGavRef.chanlocs.X]) == num_chans;
    has_spherical = length([EEGavRef.chanlocs.theta]) == num_chans;
    
    if has_cartesian & has_spherical
        % 2. Leadfield Processing
        disp([newline 'GEDAI Leadfield model: BEM warped for EEG'])
        L = load('fsavLEADFIELD_4_GEDAI.mat');
        
        % The leadfield data needs to be average referenced before interpolation
        leadfield_EEG = L.leadfield4GEDAI.EEG;
        
        % Average reference the Gain matrix (channels x sources)
        % Using non-rank-deficient average reference (to match EEG data processing)
        leadfield_EEG.data = L.leadfield4GEDAI.Gain - sum(L.leadfield4GEDAI.Gain, 1) / (size(L.leadfield4GEDAI.Gain, 1) + 1); 

        
        % 3. Interpolation and Covariance
        interpolated_EEG = interp_mont_GEDAI(leadfield_EEG, EEGavRef.chanlocs);
        refCOV = interpolated_EEG.data * interpolated_EEG.data';
        
     else
         error(['CRITICAL: Channel locations are incomplete. ' ...
               'Ensure all %d channels have X, Y, Z and spherical coordinates.'], num_chans);
    
    end


        case 'warped'
    % 1. Verification of Spatial Locations
    % We check if the number of populated X and sph_theta coordinates 
    % matches the actual number of channels.
    num_chans = length(EEGavRef.chanlocs);
    has_cartesian = length([EEGavRef.chanlocs.X]) == num_chans;
    has_spherical = length([EEGavRef.chanlocs.theta]) == num_chans;


   if has_cartesian & has_spherical
        % 2. Leadfield Processing
        disp([newline 'GEDAI Leadfield model: BEM Surface source model'])

    %  Boundary Element Method (BEM) head model based on EEGLAB/Fieldtrip source model, see https://eeglab.org/tutorials/09_source/Model_Settings.html
    [~, chanlocs_transform] = coregister(EEGin.chanlocs, 'standard_1005.elc','warp', 'auto', 'manual', 'off');
    EEGin = pop_dipfit_settings(EEGin, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat','chanfile','standard_1005.elc','coordformat','MNI','coord_transform',chanlocs_transform);
    EEGin = pop_leadfield(EEGin, 'sourcemodel','head_modelColin27_5003_Standard-10-5-Cap339.mat','sourcemodel2mni',[0 -24 -45 0 0 -1.5708 1000 1000 1000] ,'downsample',1); % Surface Colin27
    %EEGin = pop_leadfield(EEGin, 'sourcemodel','tess_cortex_mid_low_2000V.mat','sourcemodel2mni',[0 -24 -45 0 0 -1.5708 1000 1000 1000] ,'downsample',1); %  Surface ICBM152
    %EEGin = pop_leadfield(EEGin, 'sourcemodel','LORETA-Talairach-BAs.mat','sourcemodel2mni',[],'downsample',1); % Volumetric ICBM152
    
    DIPFIT_leadfield=cell2mat(EEGin.dipfit.sourcemodel.leadfield); %Gain matrix

    % Average reference the Gain matrix (channels x sources) using non-rank-deficient average reference 
    DIPFIT_leadfield = DIPFIT_leadfield- sum(DIPFIT_leadfield, 1) / (size(DIPFIT_leadfield, 1) + 1); 

    refCOV=DIPFIT_leadfield*DIPFIT_leadfield'; % gram matrix

   else
         error(['CRITICAL: Channel locations are incomplete. ' ...
               'Ensure all %d channels have X, Y, Z and spherical coordinates.'], num_chans);
   end
    end
end

% Ensure refCOV is real and perfectly symmetric to prevent eig/eigs errors
refCOV = real(refCOV);
refCOV = (refCOV + refCOV') / 2;

% --- Wavelet-based High-Pass Filtering ---
% Calculate required level to resolve lowcut_frequency
highpass_frequency=0.1;
hp_wavelet_levels = ceil(log2(EEGavRef.srate / highpass_frequency) - 1);
% Limit to maximum possible level given data length
max_possible_level = floor(log2(size(EEGin.data, 2)));
hp_wavelet_levels = min(hp_wavelet_levels, max_possible_level);
% Ensure reasonable minimum
hp_wavelet_levels = max(hp_wavelet_levels, 3);
wavelet_type = 'haar';

% Identify wavelet bands to remove based on lowcut_frequency
srate = EEGavRef.srate;
num_bands_hp = hp_wavelet_levels + 1;
upper_bounds = srate ./ (2.^(1:num_bands_hp));
bands_to_zero = find(upper_bounds <= lowcut_frequency);

% Average reference (if needed) and wavelet high-pass, into EEGavRef.data.
% hp_levels records which precision level of the high-pass cascade each piece
% ran at, so that the rebuild after the wavelet bands repeats exactly this.
tPhase = tic;
[EEGavRef.data, hp_levels] = local_prepare_avref(EEGin.data, EEGavRef.nbchan, apply_avg_ref, ...
    bands_to_zero, wavelet_type, hp_wavelet_levels, piece_bytes, []);
fprintf('[TIME] avref_hp_build %.2f [MEM] %.2f GB\n', toc(tPhase), local_mem_gb());

    % ------------------ GEDAI ------------------------------

    %% GEDAI.m never uses GEDAI_per_band's artifacts output (every call site
    %% discards it with ~), and it is a full extra copy of the band. EEGartifacts
    %% is reconstructed later as EEGavRef.data - EEGclean.data.
    %% precision = 'auto' drops to single only for bands whose epoch is longer
    %% than the channel count, where it is ~2x faster and changes the cleaned
    %% data by ~1e-5. Short-epoch bands stay in double: there the same switch
    %% moved the output by 3.3e-2, because the Gram route squares the condition
    %% number and the affected components sit near a weakly determined cut.
    %% Set to 'double' to disable, 'single' to force everywhere.
    gedai_band_opts = struct('want_artifacts', false, ...
                             'legacy_artifacts', false, ...
                             'parallel_blocks', use_block_parallel, ...
                             'precision', 'auto');
    %% Caller overrides last, so a pipeline can reach the engine's own options.
    %% The three budget fields are GEDAI's own and are not passed down.
    caller_fields = fieldnames(caller_band_opts);
    for iField = 1:numel(caller_fields)
        if ismember(caller_fields{iField}, ...
                {'memory_budget_bytes', 'materialise_bands', 'keep_highpass'})
            continue
        end
        gedai_band_opts.(caller_fields{iField}) = caller_band_opts.(caller_fields{iField});
    end

    %% The band passes run through gedai_band_engine rather than GEDAI_per_band:
    %% the engine takes its input as a source and hands back the cleaned band a
    %% stretch at a time, so the cleaned band goes straight into its destination
    %% instead of being returned whole (GEDAI_per_band is the whole-matrix
    %% wrapper of the same computation).
    disp([newline 'SENSAI threshold detection...please wait']);
    broadband_optimization_type = 'parabolic';
    broadband_maxThreshold = 12;
    broadband_source = gedai_source('matrix', EEGavRef.data);
    tPhase = tic;
    band_state = gedai_band_engine('begin', broadband_source, EEGavRef.srate, EEGavRef.chanlocs, broadband_artifact_threshold_type, broadband_epoch_size, refCOV, broadband_optimization_type, parallel, signal_type, broadband_minThreshold, broadband_maxThreshold, smoothing_window_seconds, percentile_threshold, [], gedai_band_opts);
    fprintf('[TIME] bb_threshold %.2f\n', toc(tPhase));
    tPhase = tic;
    clear broadband_source
    % Samples x channels: it becomes the level-0 wavelet approximation below,
    % whose per-channel steps want one channel per contiguous column.
    cleaned_broadband_data = zeros(size(EEGavRef.data, 2), size(EEGavRef.data, 1));
    while ~band_state.done
        [band_state, band_segment, first, last] = gedai_band_engine('step', band_state);
        cleaned_broadband_data(first:last, :) = band_segment.';
    end
    fprintf('[TIME] bb_clean %.2f [MEM] %.2f GB\n', toc(tPhase), local_mem_gb());
    [broadband_sensai, broadband_thresh, broadband_ENOVA, broadband_captured] = gedai_band_engine('finish', band_state);
    clear band_state band_segment
    % Holding this across the wavelet bands costs one group-sized double for the
    % whole stage where memory peaks; releasing it costs a second average
    % reference and high-pass at the end. Same input, same code, same values.
    keep_highpass = memory_budget > 3 * group_bytes;
    if isfield(caller_band_opts, 'keep_highpass')
        keep_highpass = logical(caller_band_opts.keep_highpass);
    end
    if keep_highpass
        fprintf('GEDAI memory: keeping the high-passed signal (no rebuild at the end).\n');
    else
        fprintf('GEDAI memory: releasing the high-passed signal, rebuilt at the end.\n');
        EEGavRef.data = [];
    end
    SENSAI_score_per_band = broadband_sensai;
    artifact_threshold_per_band = mean(broadband_thresh);
    artifact_threshold_array_per_band = {broadband_thresh};
    ENOVA_per_band = broadband_ENOVA;
    %%% Removed-field capture (issue #25, ADR 0004): grown by index exactly like
    %%% artifact_threshold_array_per_band above - band_removed_field{1} is broadband,
    %%% band_removed_field{f+1} is wavelet band f, each holding gedai_band_engine's
    %%% captured cell array (one entry per capture_windows row, empty when
    %%% capture_windows is empty).
    band_removed_field = {broadband_captured};

if broadband_only
    disp('Broadband-only mode: skipping wavelet band decomposition.');
    num_bands_to_process = 0;
    lower_frequencies = [];
    upper_frequencies = [];
    epoch_sizes_per_wavelet_band = [];
    wavelet_band_filtered_data = cleaned_broadband_data.';
    clear cleaned_broadband_data;
else

%% Second pass: Wavelet decomposition and per-band denoising
% MEMORY OPTIMIZED: Use incremental band processing instead of full decomposition
wavelet_type = 'haar';

% Calculate required number of wavelet levels to isolate lowcut_frequency
% We need: srate / 2^number_of_wavelet_bands <= lowcut_frequency
number_of_wavelet_bands = ceil(log2(EEGavRef.srate / lowcut_frequency));
% Limit to maximum possible level given data length
max_possible_level = floor(log2(size(EEGin.data, 2)));
number_of_wavelet_bands = min(number_of_wavelet_bands, max_possible_level);
% Ensure reasonable minimum
number_of_wavelet_bands = max(number_of_wavelet_bands, 6);

% OPTIMIZATION: Eliminated full wpt_EEG storage - bands will be extracted incrementally
number_of_discrete_wavelet_bands = number_of_wavelet_bands;
% Actual decomposition level needed to create number_of_discrete_wavelet_bands
actual_decomposition_level = number_of_discrete_wavelet_bands - 1;  % MODWT creates level+1 bands

% Pre-calculate center frequencies for each MRA wavelet band
srate = EEGavRef.srate;
center_frequencies = zeros(1, number_of_discrete_wavelet_bands);
lower_frequencies = zeros(1, number_of_discrete_wavelet_bands);
upper_frequencies = zeros(1, number_of_discrete_wavelet_bands);
for f = 1:number_of_discrete_wavelet_bands
    % The passband for MRA band 'f' is approx. [Fs/(2^(f+1)), Fs/(2^f)]
    lower_bound = srate / (2^(f + 1));
    upper_bound = srate / (2^f);
    center_frequencies(f) = (lower_bound + upper_bound) / 2;
    lower_frequencies(f) = lower_bound; 
    upper_frequencies(f) = upper_bound;
end


lowest_wavelet_bands_to_exclude = sum(upper_frequencies <= lowcut_frequency); 
num_bands_to_process = number_of_discrete_wavelet_bands - lowest_wavelet_bands_to_exclude;

% --- Check if data is long enough for the lowest frequency epoch size---
if num_bands_to_process > 0
    lowest_band_to_process_idx = num_bands_to_process;
    epoch_size_lowest_band = epoch_size_in_cycles / lower_frequencies(lowest_band_to_process_idx);
    required_samples = epoch_size_lowest_band * srate;

    while required_samples > size(EEGin.data, 2) && num_bands_to_process > 0
        warning('GEDAI:InsufficientData', 'EEG data length is too short for the epoch size required by the lowest frequency band (%g Hz). Increasing lowcut_frequency.', lower_frequencies(lowest_band_to_process_idx));
        lowcut_frequency = upper_frequencies(lowest_band_to_process_idx);
        lowest_wavelet_bands_to_exclude = sum(upper_frequencies <= lowcut_frequency);
        num_bands_to_process = number_of_discrete_wavelet_bands - lowest_wavelet_bands_to_exclude;
        
        lowest_band_to_process_idx = num_bands_to_process;
        epoch_size_lowest_band = epoch_size_in_cycles / lower_frequencies(lowest_band_to_process_idx);
        required_samples = epoch_size_lowest_band * srate;
    end
end

%%  Define Frequency-Dependent Epoch Sizes ---

% Calculate the ideal epoch size for each band based on the rule
epoch_sizes_per_wavelet_band = epoch_size_in_cycles ./ lower_frequencies;

% --- Display wavelet band-widths and epoch sizes ---
% disp(' ');  
left_margin = '  '; 
header1 = 'Wavelet Lower Freq (Hz)';
header2 = 'Epoch Size (s)';
str_freqs = num2str(lower_frequencies(1:num_bands_to_process)', '%.2g');
str_epochs = num2str(epoch_sizes_per_wavelet_band(1:num_bands_to_process)', '%.2g');
col1_width = max(length(header1), size(str_freqs, 2));
col2_width = max(length(header2), size(str_epochs, 2));

disp([newline 'Excluding ', num2str(lowest_wavelet_bands_to_exclude), ' wavelet bands with upper frequency < ' num2str(lowcut_frequency) ' Hz.']);


% Correct each epoch size to ensure it corresponds to an even number of samples
for f = 1:num_bands_to_process
    ideal_samples = epoch_sizes_per_wavelet_band(f) * srate;
    rounded_samples = round(ideal_samples);
    if rem(rounded_samples, 2) ~= 0
        % If odd, choose the nearest even number
        if abs(ideal_samples - (rounded_samples - 1)) < abs(ideal_samples - (rounded_samples + 1))
            final_samples = rounded_samples - 1;
        else
            final_samples = rounded_samples + 1;
        end
    else
        final_samples = rounded_samples;
    end
    epoch_sizes_per_wavelet_band(f) = final_samples / srate;
end

%% Denoise each wavelet band
% Band f is multiresolution band f of the Haar MODWT of the broadband-cleaned
% data. Its synthesis only needs the level f-1 approximation, so that
% approximation is the one array kept between bands: the band pass
% reconstructs band f from it (gedai_source 'wavelet', built straight into the
% band's working precision), and it is advanced to level f in place once band f
% is done. modwt_single_band recomputed the approximation from scratch for
% every band, next to a transposed copy of the input; the band values are the
% same, sample for sample.
% Like modwt_single_band's input, the approximation is samples x channels.
wavelet_approximation = cleaned_broadband_data;   % level 0
clear cleaned_broadband_data
[num_samples, num_channels] = size(wavelet_approximation);
success_parallel = false;

if parallel
    % MEMORY OPTIMIZED: Use 2D accumulator with correct type
    % Pre-allocate with same precision as input data
    wavelet_band_filtered_data = zeros(num_channels, num_samples, 'like', wavelet_approximation);
    % The band-parallel loop gives every worker the whole input, as before.
    unfiltered_data = wavelet_approximation;
    try
        temp_sensai_scores = zeros(1, num_bands_to_process);
        temp_thresholds = zeros(1, num_bands_to_process);
        temp_thresholds_arrays = cell(1, num_bands_to_process);
        temp_enova_scores = zeros(1, num_bands_to_process);
        
        % MEMORY OPTIMIZED: Incremental band extraction in parallel
        parfor f = 1:num_bands_to_process
            % Extract single band on-the-fly (no full wpt_EEG storage)
            wavelet_data_band = modwt_single_band(unfiltered_data, wavelet_type, actual_decomposition_level, f)';
            
            current_epoch_size = epoch_sizes_per_wavelet_band(f);
            
            % Determine minThreshold based on signal type and frequency
            current_center_freq = center_frequencies(f);
            current_minThreshold = 0;
            if (current_center_freq >= 0.5 && current_center_freq <= 60)
                current_minThreshold = -6;
            end

            try
                 [cleaned_band_data, ~, temp_sensai, temp_thresh, temp_enova_val] = GEDAI_per_band(wavelet_data_band, srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold, [], smoothing_window_seconds, percentile_threshold, [], gedai_band_opts);
            catch ME
                 % If OOM or other memory error, try single precision
                 warning('GEDAI_per_band failed for band %d: %s. Retrying with single precision...', f, ME.message);
                 [cleaned_band_data, ~, temp_sensai, temp_thresh, temp_enova_val] = GEDAI_per_band(single(wavelet_data_band), srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold, [], smoothing_window_seconds, percentile_threshold, [], gedai_band_opts);
            end
            
            % RAM OPTIMIZATION: Accumulate directly using a reduction variable (avoids massive cell array copies)
            wavelet_band_filtered_data = wavelet_band_filtered_data + cleaned_band_data;
            temp_sensai_scores(f) = temp_sensai;
            temp_thresholds(f) = mean(temp_thresh);
            temp_thresholds_arrays{f} = temp_thresh;
            temp_enova_scores(f) = temp_enova_val;
        end
        
        SENSAI_score_per_band = [SENSAI_score_per_band, temp_sensai_scores];
        artifact_threshold_per_band = [artifact_threshold_per_band, temp_thresholds];
        artifact_threshold_array_per_band = [artifact_threshold_array_per_band, temp_thresholds_arrays];
        ENOVA_per_band = [ENOVA_per_band, temp_enova_scores];
        success_parallel = true;
    catch
        warning('Parallel processing failed: %s. Switching to double precision non-parallel processing.');
    end
    clear unfiltered_data
end

if ~parallel || ~success_parallel
    if parallel && ~success_parallel
         disp('Executing fallback: Double Precision Non-Parallel Processing...');
    end

    inv_sqrt2 = 1 / sqrt(2);
    channels_per_piece = max(1, floor(piece_bytes / (8 * num_samples)));
    % Channels x samples, the shape the engine emits and the shape the output
    % wants. Held as samples x channels before, which cost a transpose of every
    % emitted stretch on the way in and one of the whole array on the way out;
    % the values are the same either way, and a sample's channels are now
    % contiguous.
    band_sum = zeros(num_channels, num_samples);
    % A band read from the approximation is re-synthesised on every read, so
    % synthesising it once into a buffer and reading slices looks like an obvious
    % win. Measured on sub-drop0001 ses-t1 (243 ch, four stage groups, whole
    % recording) it is not: 36.38 min with the buffer against 35.90 min without,
    % output byte-identical either way. Most bands here have fewer than 500 band
    % epochs and take the engine's legacy path, which reads the band once anyway,
    % so the buffer only adds a synthesis and a band-sized array. Off by default;
    % materialise_bands turns it on for a band mix where the reads do repeat.
    materialise_bands = false;
    if isfield(caller_band_opts, 'materialise_bands')
        materialise_bands = logical(caller_band_opts.materialise_bands);
    end
    if materialise_bands
        fprintf('GEDAI memory: materialising each wavelet band once.\n');
    else
        fprintf('GEDAI memory: reading each wavelet band from the approximation.\n');
    end
    for f = 1:num_bands_to_process
        current_epoch_size = epoch_sizes_per_wavelet_band(f);

        % Determine minThreshold based on signal type and frequency
        current_center_freq = center_frequencies(f);
        current_minThreshold = 0;
        if (current_center_freq >= 0.5 && current_center_freq <= 60)
            current_minThreshold = -6;
        end

        disp(sprintf('processing wavelet band = %d/%d (%.2g - %.2g Hz)', f, num_bands_to_process, lower_frequencies(f), upper_frequencies(f)))
        tPhase = tic;
        if materialise_bands
            band_class = local_band_class(gedai_band_opts.precision, ...
                round(srate * current_epoch_size), num_channels);
            band_matrix = gedai_wavelet_band(wavelet_approximation, f, ...
                actual_decomposition_level, band_class);
            band_source = gedai_source('matrix_tc', band_matrix);
        else
            band_source = gedai_source('wavelet', wavelet_approximation, f, actual_decomposition_level);
        end
        tThisSource = toc(tPhase);

        % A band that fails before it has written any output is retried in single
        % precision, as before. Once output has gone into the accumulator a retry
        % would count part of the band twice, so the error is passed on instead.
        % (The whole-matrix version fell back to redoing every band in single on
        % top of an accumulator that already held the finished ones.)
        output_started = false;
        for attempt = 1:2
            try
                tPhase = tic;
                band_state = gedai_band_engine('begin', band_source, srate, EEGavRef.chanlocs, artifact_threshold_type, current_epoch_size, refCOV, 'parabolic', false, signal_type, current_minThreshold, [], smoothing_window_seconds, percentile_threshold, [], gedai_band_opts);
                tThisBegin = toc(tPhase);
                tPhase = tic; tThisAcc = 0;
                while ~band_state.done
                    [band_state, band_segment, first, last] = gedai_band_engine('step', band_state);
                    output_started = true;
                    % MEMORY OPTIMIZED: Accumulate directly into 2D array
                    tAcc = tic;
                    band_sum(:, first:last) = band_sum(:, first:last) + double(band_segment);
                    tThisAcc = tThisAcc + toc(tAcc);
                end
                fprintf('[TIME] band %d source %.2f threshold %.2f clean %.2f accumulate %.2f [MEM] %.2f GB\n', ...
                    f, tThisSource, tThisBegin, toc(tPhase) - tThisAcc, tThisAcc, local_mem_gb());
                break
            catch ME
                if attempt == 2 || output_started
                    rethrow(ME);
                end
                warning('GEDAI_per_band failed for band %d: %s. Retrying with single precision...', f, ME.message);
                band_source.inputClass = 'single';
                if materialise_bands
                    band_matrix = single(band_matrix);
                    band_source = gedai_source('matrix_tc', band_matrix);
                end
            end
        end
        [sensai_val, thresh_val, enova_val, band_captured_f] = gedai_band_engine('finish', band_state);
        clear band_source band_state band_segment band_matrix

        SENSAI_score_per_band(f+1) = sensai_val;
        artifact_threshold_per_band(f+1) = mean(thresh_val);
        artifact_threshold_array_per_band{f+1} = thresh_val;
        ENOVA_per_band(f+1) = enova_val;
        band_removed_field{f+1} = band_captured_f;

        % Advance the approximation to level f, in place and a few channels at a
        % time: exactly modwt_single_band's forward step, which shifts along time
        % only. (band_source was cleared first so this does not copy the array.)
        if f < num_bands_to_process
            shift = 2^(f-1);
            for c0 = 1:channels_per_piece:num_channels
                chans = c0:min(num_channels, c0 + channels_per_piece - 1);
                wavelet_approximation(:, chans) = (wavelet_approximation(:, chans) + circshift(wavelet_approximation(:, chans), shift, 1)) * inv_sqrt2;
            end
        end
    end
    clear wavelet_approximation
    wavelet_band_filtered_data = band_sum;
    clear band_sum
end

clear wavelet_approximation

end % broadband_only

%%% Pad band_removed_field to one entry per processed band (issue #25, ADR 0004): the
%%% classic band-parallel loop above (parallel = true, not 'blocks') does not populate
%%% per-band entries past the broadband one - capture_windows is guarded to be empty
%%% whenever that loop runs, so the padding is always {} there, never a silent gap.
if numel(band_removed_field) < num_bands_to_process + 1
    band_removed_field(numel(band_removed_field) + 1 : num_bands_to_process + 1) = {cell(0, 1)};
end

%% Finalization: Reconstruct EEG and calculate final scores
% MEMORY OPTIMIZED: Data already accumulated in 2D array, no summation needed
% Rebuild the average-referenced, high-passed input if it was released after the
% broadband pass: same input, same code, same high-pass precision per piece. When
% the budget allowed it to be held, it is already here and this is skipped.
if isempty(EEGavRef.data)
    EEGavRef.data = local_prepare_avref(EEGin.data, EEGavRef.nbchan, apply_avg_ref, ...
        bands_to_zero, wavelet_type, hp_wavelet_levels, piece_bytes, hp_levels);
end
EEGclean = EEGavRef;
EEGclean.data = wavelet_band_filtered_data;  % Already accumulated
clear wavelet_band_filtered_data
% Create artifact structure
% It is a third copy of the recording, so it is only formed when something reads
% it: the output argument, SENSAI scoring (and the epoch rejection that depends
% on it), the visualisation, or restoring epoched input. ENOVA per channel takes
% the difference one second at a time instead.
need_artifacts = nargout > 1 || compute_SENSAI || visualize_artifacts || is_epoched;
if need_artifacts
    EEGartifacts = EEGclean;
    EEGartifacts.data = EEGavRef.data(:, 1:size(EEGclean.data, 2)) - EEGclean.data;
else
    EEGartifacts = [];
end

% Calculate composite SENSAI score for epoch rejection
noise_multiplier = 1;
sensai_epoch_size = 1;
if ENOVA_threshold_per_epoch < inf & compute_SENSAI
    tSensai1 = tic;
    [SENSAI_score, ~, ~, mean_ENOVA, ENOVA_per_epoch_internal] = SENSAI_basic(double(EEGclean.data), double(EEGartifacts.data), EEGavRef.srate, sensai_epoch_size, refCOV, noise_multiplier, signal_type);
    fprintf('SENSAI_basic (epoch scoring): %.2f s\n', toc(tSensai1));
    if ~isempty(precomputed_ENOVA_per_epoch)
        ENOVA_per_epoch = precomputed_ENOVA_per_epoch;
    else
        ENOVA_per_epoch = ENOVA_per_epoch_internal;
    end
else
    ENOVA_per_epoch = [];
end

% Store original epoch count for rejection statistics
epoch_samples_for_count = round(sensai_epoch_size * EEGavRef.srate);
original_total_epochs = max(length(ENOVA_per_epoch), floor(size(EEGclean.data, 2) / epoch_samples_for_count));

% Calculate ENOVA per channel as a mean across epochs
epoch_samples = round(sensai_epoch_size * EEGavRef.srate);
pnts_total = size(EEGclean.data, 2);
num_epochs_ch = floor(pnts_total / epoch_samples);

if num_epochs_ch > 0
    % One epoch at a time, straight from EEGavRef and EEGclean: the artifact
    % signal of an epoch is the same difference EEGartifacts holds, and var()
    % sees the same channels x samples matrix as when it was cut from the
    % truncated, reshaped copies of both arrays.
    enova_ch_epochs = zeros(size(EEGavRef.data, 1), num_epochs_ch);
    for ep = 1:num_epochs_ch
        cols = (ep-1)*epoch_samples+1 : ep*epoch_samples;
        orig_epoch = EEGavRef.data(:, cols);
        var_orig = var(orig_epoch, 0, 2);
        var_noise = var(orig_epoch - EEGclean.data(:, cols), 0, 2);
        enova_ch_epochs(:, ep) = var_noise ./ var_orig;
    end
    clear orig_epoch
    ENOVA_per_channel = mean(enova_ch_epochs, 2);
else
    var_artifacts_per_channel = var(EEGavRef.data(:, 1:size(EEGclean.data, 2)) - EEGclean.data, 0, 2);
    var_original_per_channel = var(EEGavRef.data(:, 1:size(EEGclean.data, 2)), 0, 2);
    ENOVA_per_channel = var_artifacts_per_channel ./ var_original_per_channel;
end

epochs_to_remove = find(ENOVA_per_epoch > ENOVA_threshold_per_epoch);
regions = [];
if ~isempty(epochs_to_remove)
    epoch_samples = round(sensai_epoch_size * EEGavRef.srate);
    regions = zeros(length(epochs_to_remove), 2);
    for i = 1:length(epochs_to_remove)
        epoch = epochs_to_remove(i);
        start_sample = (epoch - 1) * epoch_samples + 1;
        end_sample = epoch * epoch_samples;
        if end_sample > size(EEGclean.data, 2)
            end_sample = size(EEGclean.data, 2);
        end
        regions(i,:) = [start_sample end_sample];
    end
end

tEnd = toc(tStart);


% Generate command history
if ~ischar(ref_matrix_type)
    ref_matrix_type = 'custom';
end
com = sprintf('EEG = GEDAI(EEG, ''%s'', ''%s'', %s,  %s, ''%s'', %d,  %d, %s, %s, ''%s'');', ...
    artifact_threshold_type, broadband_artifact_threshold_type, num2str(epoch_size_in_cycles), num2str(lowcut_frequency), ref_matrix_type, parallel, visualize_artifacts, num2str(ENOVA_threshold_per_epoch), num2str(ENOVA_threshold_per_channel), signal_type);

if visualize_artifacts
    EEGclean_for_vis = EEGclean;
    if ~isempty(regions)
        clean_sample_mask = true(1, EEGclean_for_vis.pnts);
        for i = 1:size(regions, 1)
            clean_sample_mask(regions(i,1):regions(i,2)) = false;
        end
        EEGclean_for_vis.etc.clean_sample_mask = clean_sample_mask;
        EEGclean_for_vis.data = EEGclean_for_vis.data(:, clean_sample_mask);
        EEGclean_for_vis.pnts = size(EEGclean_for_vis.data, 2);
    end
    vis_artifacts(EEGclean_for_vis, EEGavRef, 'ScaleBy', 'noscale', 'YScaling', 3*mad(EEGavRef.data(:)));
end

if ~isempty(regions)
    disp([newline 'Removing bad epochs...']);
    
    % Manual implementation of eeg_eegrej to avoid eeg_checkset issues
    samples_to_keep = true(1, EEGclean.pnts);
    for i = 1:size(regions, 1)
        start_idx = round(regions(i,1));
        end_idx = round(regions(i,2));
        if start_idx > 0 && end_idx <= EEGclean.pnts
            samples_to_keep(start_idx:end_idx) = false;
        end
    end

    % --- Apply windowing/tapering to smooth discontinuities at the seams after epoch rejection ---
    taper_duration = 0.05; % 50 ms
    taper_points = round(taper_duration * EEGclean.srate);
    % Create a cosine taper (half-Hanning)
    % We want 0 to 1 over 'taper_points'. 
    % Hanning is (1 - cos(phi))/2. 
    % phi=0 -> 0. phi=pi -> 1.
    taper_phase = linspace(0, pi, taper_points);
    taper_attack = (1 - cos(taper_phase)) / 2; % Rise 0 to 1
    taper_decay = fliplr(taper_attack);        % Fall 1 to 0

    % Find transitions
    % diff = -1: Keep -> Reject (End of valid segment) -> Apply Decay
    decay_indices = find(diff(samples_to_keep) == -1);
    
    % diff = 1: Reject -> Keep (Start of valid segment) -> Apply Attack
    attack_indices = find(diff(samples_to_keep) == 1) + 1;
    
    % Apply Decay (Fade Out)
    for idx = decay_indices
        s_start = max(1, idx - taper_points + 1);
        s_end = idx;
        len = s_end - s_start + 1;
        
        current_taper = taper_decay(end-len+1:end); % Match length
        EEGclean.data(:, s_start:s_end) = EEGclean.data(:, s_start:s_end) .* current_taper; 
        EEGartifacts.data(:, s_start:s_end) = EEGartifacts.data(:, s_start:s_end) .* current_taper;
    end
    
    % Apply Attack (Fade In)
    for idx = attack_indices
        s_start = idx;
        s_end = min(EEGclean.pnts, idx + taper_points - 1);
        len = s_end - s_start + 1;
        
        current_taper = taper_attack(1:len);
        EEGclean.data(:, s_start:s_end) = EEGclean.data(:, s_start:s_end) .* current_taper;
        EEGartifacts.data(:, s_start:s_end) = EEGartifacts.data(:, s_start:s_end) .* current_taper;
    end
    % -----------------------------------------------------------------------

    % Apply mask to EEGclean
    EEGclean.data = EEGclean.data(:, samples_to_keep);
    EEGclean.pnts = size(EEGclean.data, 2);
    EEGclean.xmax = EEGclean.xmin + (EEGclean.pnts-1)/EEGclean.srate;
    if EEGclean.pnts > 1
        EEGclean.times = linspace(EEGclean.xmin*1000, EEGclean.xmax*1000, EEGclean.pnts);
    else
        EEGclean.times = EEGclean.xmin*1000;
    end
    
    % Apply mask to EEGartifacts
    EEGartifacts.data = EEGartifacts.data(:, samples_to_keep);
    EEGartifacts.pnts = size(EEGartifacts.data, 2);
    EEGartifacts.xmax = EEGartifacts.xmin + (EEGartifacts.pnts-1)/EEGartifacts.srate;
    if EEGartifacts.pnts > 1
        EEGartifacts.times = linspace(EEGartifacts.xmin*1000, EEGartifacts.xmax*1000, EEGartifacts.pnts);
    else
        EEGartifacts.times = EEGartifacts.xmin*1000;
    end
    
    % --- Update event latencies ---
    % Replicate eeg_eegrej logic: shift events and remove those lying within rejected regions
    if isfield(EEGclean, 'event') && ~isempty(EEGclean.event) && isfield(EEGclean.event, 'latency')
        eventLatencies = [EEGclean.event.latency];
        oriEventLatencies = eventLatencies;
        rmEvent = [];
        
        % Ensure regions are sorted
        regions_sorted = sortrows(sort(regions, 2));
        
        for iReg = 1:size(regions_sorted, 1)
            % Find events within the current rejected region
            reject_idx = find(oriEventLatencies >= regions_sorted(iReg,1) & oriEventLatencies <= regions_sorted(iReg,2));
            rmEvent = [rmEvent reject_idx];
            
            % Shift events occurring after the start of this rejected region
            shift_amount = regions_sorted(iReg,2) - regions_sorted(iReg,1) + 1;
            idx_to_shift = find(oriEventLatencies > regions_sorted(iReg,1));
            eventLatencies(idx_to_shift) = eventLatencies(idx_to_shift) - shift_amount;
        end
        
        for iEvent = 1:length(EEGclean.event)
            EEGclean.event(iEvent).latency = eventLatencies(iEvent);
        end
        EEGclean.event(rmEvent) = [];
    end

    if isfield(EEGartifacts, 'event')
        EEGartifacts.event = EEGclean.event;
    end
end

% Calculate final SENSAI score (after potential epoch rejection)

if compute_SENSAI
    tSensai2 = tic;
    [SENSAI_score, ~, ~, mean_ENOVA, ENOVA_per_epoch] = SENSAI_basic(double(EEGclean.data), double(EEGartifacts.data), EEGavRef.srate, 1, refCOV, noise_multiplier, signal_type);
    fprintf('SENSAI_basic (final scoring): %.2f s\n', toc(tSensai2));
else
    SENSAI_score = NaN;
    mean_ENOVA   = NaN;
    ENOVA_per_epoch = [];
end

% disp([newline 'SENSAI score: ' num2str(round(SENSAI_score, 2, 'significant'))]);
% disp(['Mean ENOVA: ' num2str(round(mean_ENOVA, 2, 'significant'))]);

% Use original epoch count for rejection statistics (before rejection)
num_rejected = length(epochs_to_remove);
if original_total_epochs > 0
    percentage_rejected = (num_rejected / original_total_epochs) * 100;
else
    percentage_rejected = 0;
end
% disp(['Bad epochs rejected: ' num2str(round(percentage_rejected,1)) ' % (' num2str(num_rejected) ' out of ' num2str(original_total_epochs) ' epochs)']);

% --- Summarized Output Table (including ENOVA) ---
disp(' '); 
left_margin = '  '; 
header1 = 'Wavelet Lower Freq (Hz)';
header2 = 'Epoch Size (s)';
header3 = 'ENOVA (%)';

% Combine frequencies and epochs for display, including Broadband
% Broadband is index 1 in the arrays, usually displayed first
% We can label frequency as "Broadband" or Inf/NaN
freq_str_cell = cell(1, num_bands_to_process + 1);
freq_str_cell{1} = 'Broadband';
for i = 1:num_bands_to_process
    freq_str_cell{i+1} = [num2str(lower_frequencies(i), '%.2g') ' Hz'];
end

epoch_str_cell = cell(1, num_bands_to_process + 1);
epoch_str_cell{1} = [num2str(broadband_epoch_size, '%.2g') ' s'];
for i = 1:num_bands_to_process
    epoch_str_cell{i+1} = [num2str(epoch_sizes_per_wavelet_band(i), '%.2g') ' s'];
end

enova_str_cell = cell(1, num_bands_to_process + 1);
% ENOVA_per_band contains [Broadband, Band1, Band2, ...] if processed sequentially
% Ensure correct indexing
for i = 1:length(ENOVA_per_band)
    enova_str_cell{i} = [num2str(round(ENOVA_per_band(i) * 100), '%.0f') ' %'];
end

% Determine column widths
max_freq_width = max(cellfun(@length, freq_str_cell));
col1_width = max(length(header1), max_freq_width) + 2; % Add padding
max_epoch_width = max(cellfun(@length, epoch_str_cell));
col2_width = max(length(header2), max_epoch_width) + 2; % Add padding
max_enova_width = max(cellfun(@length, enova_str_cell));
col3_width = max(length(header3), max_enova_width) + 2; % Add padding

% Centering helper function
center_text = @(str, width) [repmat(' ', 1, floor((width-length(str))/2)), str, repmat(' ', 1, ceil((width-length(str))/2))];

fprintf('%s%s | %s | %s\n', left_margin, center_text(header1, col1_width), center_text(header2, col2_width), center_text(header3, col3_width));
fprintf('%s%s-|-%s-|-%s\n', left_margin, repmat('-', 1, col1_width), repmat('-', 1, col2_width), repmat('-', 1, col3_width));

for i = 1:length(freq_str_cell)
    fprintf('%s%s | %s | %s\n', left_margin, center_text(freq_str_cell{i}, col1_width), center_text(epoch_str_cell{i}, col2_width), center_text(enova_str_cell{i}, col3_width));
end
disp(' ');

disp([newline 'SENSAI score: ' num2str(round(SENSAI_score, 2, 'significant'))]);
disp(['Mean ENOVA: ' num2str(round(mean_ENOVA*100, 2, 'significant')) ' %']);
disp(['Bad epochs rejected: ' num2str(round(percentage_rejected,1)) ' % (' num2str(num_rejected) ' out of ' num2str(original_total_epochs) ' epochs)']);
disp(['Elapsed time: ' num2str(round(tEnd, 2, 'significant')) ' seconds' newline]);

if smoothing_window_seconds ~= Inf
    disp('Note: Threshold successfully adapted to non-stationarities over time.');

disp(' ');

if visualize_artifacts
    plot_title = ['GEDAI Sliding Thresholds (' artifact_threshold_type ' | Window: ' num2str(smoothing_window_seconds) ' s | SENSAI: ' num2str(round(SENSAI_score, 1)) '%)'];
    figure('Color', 'w', 'Name', plot_title);
    num_plots = length(artifact_threshold_array_per_band);
    
    % Create a tiled layout based on the number of plots (max 3 columns)
    num_cols = min(num_plots, 3);
    num_rows = ceil(num_plots / num_cols);
    tiledlayout(num_rows, num_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(plot_title, 'FontSize', 12, 'FontWeight', 'bold');
    
    % Distinct perceptually-spaced colours for each band
    band_colors = turbo(max(num_plots, 1));
    
    for i = 1:num_plots
        nexttile;
        thresh_array = artifact_threshold_array_per_band{i};
        
        % Determine correct epoch size for accurate time axis
        if i == 1
            current_epoch_size = broadband_epoch_size;
        else
            current_epoch_size = epoch_sizes_per_wavelet_band(i-1);
        end
        
        time_axis_minutes = (1:length(thresh_array)) * current_epoch_size / 60;
        plot(time_axis_minutes, thresh_array, '-', 'Color', band_colors(i,:), 'LineWidth', 2);
        
        title(freq_str_cell{i}, 'FontSize', 12);
        
        % Only label x-axis on the bottom row to save space
        if i > num_plots - num_cols
            xlabel('Time (Minutes)', 'FontSize', 10);
        end
        ylabel('Threshold', 'FontSize', 10);
        grid on;
        
        % Fixed y-axis scale across all bands for easy comparison
        ylim([-1.9, 10]);
    end
end
end

% Store GEDAI variables in EEG.etc.GEDAI
EEGclean.etc.GEDAI.SENSAI_score = SENSAI_score;
EEGclean.etc.GEDAI.SENSAI_score_per_band = SENSAI_score_per_band;
EEGclean.etc.GEDAI.artifact_threshold_per_band = artifact_threshold_per_band;
EEGclean.etc.GEDAI.artifact_threshold_array_per_band = artifact_threshold_array_per_band;
EEGclean.etc.GEDAI.freq_str_cell = freq_str_cell;
EEGclean.etc.GEDAI.epoch_sizes_per_wavelet_band = epoch_sizes_per_wavelet_band;
EEGclean.etc.GEDAI.broadband_epoch_size = broadband_epoch_size;
EEGclean.etc.GEDAI.mean_ENOVA = mean_ENOVA;
EEGclean.etc.GEDAI.ENOVA_per_band = ENOVA_per_band;
EEGclean.etc.GEDAI.ENOVA_per_epoch = ENOVA_per_epoch;
EEGclean.etc.GEDAI.ENOVA_per_channel = ENOVA_per_channel;
EEGclean.etc.GEDAI.epochs_rejected = num_rejected;
EEGclean.etc.GEDAI.total_epochs = original_total_epochs;
EEGclean.etc.GEDAI.percentage_rejected = percentage_rejected;
if exist('samples_to_keep', 'var')
    EEGclean.etc.GEDAI.samples_to_keep = samples_to_keep;
else
    EEGclean.etc.GEDAI.samples_to_keep = true(1, original_total_epochs * round(sensai_epoch_size * EEGavRef.srate)); 
    % Note: The above calculation might be slightly off if rounding happened differently for 'pnts'.
    % Safer to use current pnts if no rejection happened:
    EEGclean.etc.GEDAI.samples_to_keep = true(1, size(EEGclean.data, 2));
end


    % --- Manifold Classification (Broadband) BEFORE & AFTER Cleaning ---
    % Uses 50% overlapping 1-second epochs for denser coverage in the scatter plot
    if visualize_artifacts && ~isempty(refCOV)
        % Ensure visualization uses the same PC count as the SENSAI scoring logic
        if strcmpi(signal_type, 'meg'), vis_pcs = 4; else, vis_pcs = 3; end
        
        visualization_metrics = SENSAI_visualization(EEGavRef, EEGclean, EEGartifacts, refCOV, sensai_epoch_size, signal_type, vis_pcs, artifact_threshold_type, smoothing_window_seconds, SENSAI_score, mean_ENOVA);
        
        % Store metrics in EEG.etc.GEDAI
        EEGclean.etc.GEDAI.visualization_metrics = visualization_metrics;
    end

% Add command history to EEGLAB structure
if exist('eegh', 'file')
    EEGclean = eegh(com, EEGclean);
end

% -- Restore Epoched Data Structure if needed --
if is_epoched
    if size(EEGclean.data, 2) == size(original_EEG.data, 2) * size(original_EEG.data, 3)
        disp('Converting continuous data back to epoched structure...');
        % Restore structure for clean
        EEGclean_epoched = original_EEG;
        EEGclean_epoched.data = reshape(EEGclean.data, size(EEGclean_epoched.data));
        EEGclean_epoched.history = EEGclean.history;
        if isfield(EEGclean, 'etc')
            EEGclean_epoched.etc = EEGclean.etc;
        end
        EEGclean = EEGclean_epoched;
        
        % Restore structure for artifacts
        EEGartifacts_epoched = original_EEG;
        EEGartifacts_epoched.data = reshape(EEGartifacts.data, size(EEGartifacts_epoched.data));
        EEGartifacts = EEGartifacts_epoched;
    else
        warning('Data length changed (e.g., due to epoch rejection). Cannot cleanly restore 3D structure. Returning data as continuous.');
    end
end

bandRemovedField = band_removed_field;

end

% =========================================================================
function gb = local_mem_gb()
%LOCAL_MEM_GB  What this MATLAB process currently holds, in GB.
%   Measured inside the process, because the machine is shared: an external
%   working-set reading cannot tell this run's arrays from a neighbour's.
gb = NaN;
try
    if ispc
        m = memory;
        gb = m.MemUsedMATLAB / 2^30;
    else
        txt = fileread(sprintf('/proc/%d/statm', feature('getpid')));
        pages = sscanf(txt, '%f');
        gb = pages(2) * 4096 / 2^30;   % resident pages
    end
catch
end
end
% =========================================================================
function budget = local_memory_budget(caller_band_opts)
%LOCAL_MEMORY_BUDGET  Bytes this run may allocate on top of what it holds.
%   memory_budget_bytes in the caller's band-options struct wins, which is how a
%   64 GB machine's behaviour is reproduced and measured on a larger one. Without
%   it the machine is asked: MEMORY on Windows, /proc/meminfo elsewhere. When
%   neither answers, the budget is 0, i.e. the low-memory path, because
%   over-committing is the failure that loses a whole recording.
if isfield(caller_band_opts, 'memory_budget_bytes') ...
        && ~isempty(caller_band_opts.memory_budget_bytes)
    budget = double(caller_band_opts.memory_budget_bytes);
    return
end
budget = 0;
try
    if ispc
        m = memory;
        budget = m.MaxPossibleArrayBytes;
    else
        txt = fileread('/proc/meminfo');
        tok = regexp(txt, 'MemAvailable:\s+(\d+) kB', 'tokens', 'once');
        if ~isempty(tok)
            budget = str2double(tok{1}) * 1024;
        end
    end
catch
    budget = 0;
end
end

% =========================================================================
function cls = local_band_class(precision, Te, N)
%LOCAL_BAND_CLASS  The class a band pass will work in, for pre-building its data.
%   Mirrors the precision policy in gedai_band_engine: 'auto' uses single only
%   where the band epoch is longer than the channel count. A band buffer built in
%   the wrong class would be cast on every read, so this has to agree with the
%   engine.
switch lower(precision)
    case 'single'
        cls = 'single';
    case 'auto'
        if Te > N
            cls = 'single';
        else
            cls = 'double';
        end
    otherwise
        cls = 'double';
end
end
% =========================================================================
function max_offset = local_max_reference_offset(input, piece_bytes)
%LOCAL_MAX_REFERENCE_OFFSET  max(abs(sum(double(input), 1) / (N + 1))), a block of
%   samples at a time. Each column sum only involves its own column, so the
%   result is the same as over the whole double-converted recording.
N = size(input, 1);
samples_per_piece = max(1, floor(piece_bytes / (8 * N)));
max_offset = NaN;          % max() skips NaN, as it did over the whole array
for s0 = 1:samples_per_piece:size(input, 2)
    cols = s0:min(size(input, 2), s0 + samples_per_piece - 1);
    max_offset = max(max_offset, max(abs(sum(double(input(:, cols)), 1) / (N + 1))));
end
end

% =========================================================================
function [data, hp_levels] = local_prepare_avref(input, nbchan, apply_avg_ref, bands_to_zero, ...
    wavelet_type, hp_wavelet_levels, piece_bytes, hp_levels)
%LOCAL_PREPARE_AVREF  Double-precision input, average referenced and high-passed.
%
%   The same arithmetic as converting the whole recording to double, calling
%   GEDAI_nonRankDeficientAveRef and removing the low wavelet bands with
%   modwt_single_band on the transposed recording, done in place and in pieces:
%   the reference of a sample only involves its own column, and the MODWT only
%   shifts along time, so every value is unchanged. Only the result and a piece's
%   temporaries are held, instead of several copies of the recording.
%
%   High-pass precision, as before: GPU double -> GPU single -> CPU double ->
%   CPU single, falling to the next level when one throws. hp_levels (one entry
%   per channel piece) records the level each piece ran at; pass it back in to
%   rebuild the signal with the same levels.
data = double(input);

if apply_avg_ref
    samples_per_piece = max(1, floor(piece_bytes / (8 * size(data, 1))));
    for s0 = 1:samples_per_piece:size(data, 2)
        cols = s0:min(size(data, 2), s0 + samples_per_piece - 1);
        data(:, cols) = data(:, cols) - sum(data(:, cols), 1) / (nbchan + 1);
    end
end

if ~isempty(bands_to_zero)
    % The MODWT runs on the transposed recording (samples x channels), as before,
    % where one channel is one contiguous column.
    warning('off');
    data = data.';
    channels_per_piece = max(1, floor(piece_bytes / (8 * size(data, 1))));
    piece_starts = 1:channels_per_piece:size(data, 2);
    replay = ~isempty(hp_levels);
    if ~replay
        hp_levels = zeros(1, numel(piece_starts));
    end
    level = 0;
    for p = 1:numel(piece_starts)
        chans = piece_starts(p):min(size(data, 2), piece_starts(p) + channels_per_piece - 1);
        if replay
            level = hp_levels(p);
        end
        [low_freq_noise, level] = local_hp_noise(data(:, chans), bands_to_zero, ...
            wavelet_type, hp_wavelet_levels, level);
        hp_levels(p) = level;
        data(:, chans) = data(:, chans) - low_freq_noise;
    end
    clear low_freq_noise
    data = data.';
end
end

% =========================================================================
function [low_freq_noise, level] = local_hp_noise(data, bands_to_zero, wavelet_type, hp_wavelet_levels, level)
%LOCAL_HP_NOISE  Wavelet high-pass noise of a samples x channels piece, in double.
%   level is the position in the precision cascade, kept across pieces:
%   0 not started, 1 GPU double, 2 GPU single, 3 CPU double, 4 CPU single.
%   A level that throws is left for the next one; CPU single has no fallback.
if level == 0
    if gpuDeviceCount > 0
        level = 1;
        disp('Attempting GPU processing (Double Precision)...');
        parallel.gpu.enableCUDAForwardCompatibility(true)
    else
        level = 3;
        disp('Attempting CPU processing (Double Precision)...');
    end
end
while true
    try
        switch level
            case 1
                data_gpu = gpuArray(data);
                low_freq_noise_gpu = zeros(size(data_gpu), 'like', data_gpu);
                for b = 1:length(bands_to_zero)
                    low_freq_noise_gpu = low_freq_noise_gpu + modwt_single_band(data_gpu, wavelet_type, hp_wavelet_levels, bands_to_zero(b));
                end
                low_freq_noise = gather(low_freq_noise_gpu);
            case 2
                data_gpu = gpuArray(single(data));
                low_freq_noise_gpu = zeros(size(data_gpu), 'like', data_gpu);
                for b = 1:length(bands_to_zero)
                    low_freq_noise_gpu = low_freq_noise_gpu + modwt_single_band(data_gpu, wavelet_type, hp_wavelet_levels, bands_to_zero(b));
                end
                low_freq_noise = double(gather(low_freq_noise_gpu));
            case 3
                low_freq_noise = zeros(size(data), 'like', data);
                for b = 1:length(bands_to_zero)
                    low_freq_noise = low_freq_noise + modwt_single_band(data, wavelet_type, hp_wavelet_levels, bands_to_zero(b));
                end
            case 4
                data_single = single(data);
                low_freq_noise = zeros(size(data_single), 'like', data_single);
                for b = 1:length(bands_to_zero)
                    low_freq_noise = low_freq_noise + modwt_single_band(data_single, wavelet_type, hp_wavelet_levels, bands_to_zero(b));
                end
                low_freq_noise = double(low_freq_noise);
            otherwise
                error('GEDAI:hpLevel', 'Unknown high-pass precision level %d.', level);
        end
        return
    catch ME
        switch level
            case 1
                warning('GPU (Double) failed. Attempting GPU (Single Precision)...');
            case 2
                warning('GPU (Single) failed. Falling back to CPU.');
                disp('Attempting CPU processing (Double Precision)...');
            case 3
                warning('CPU (Double) failed. Attempting CPU (Single Precision)...');
            otherwise
                rethrow(ME);
        end
        level = level + 1;
    end
end
end
