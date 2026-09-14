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

function [cleaned_data, artifacts_data, SENSAI_score, artifact_threshold_out, ENOVA] = GEDAI_per_band(eeg_data, srate, chanlocs, artifact_threshold_type, epoch_size, refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold, smoothing_window_seconds, percentile_threshold, rank_truncation, opts)
%   rank_truncation - logical, default true. When an epoch is shorter than the
%                     channel count its covariance is rank deficient, and the
%                     GEVD is truncated to the supported subspace. This is the
%                     large memory and speed win on the fast bands. Set false
%                     to force the full N x N decomposition, which is the
%                     reference behaviour; see the note at local_gevd in
%                     gedai_band_engine.
%
%   opts            - struct of options.
%                     .want_artifacts   return artifacts_data (default true).
%                     .legacy_artifacts return them on the legacy path even
%                                       when want_artifacts is false (default
%                                       true, as this function always did).
%                     .precision        'double' (default), 'single', or 'auto'
%                                       (single only when the epoch is longer
%                                       than the channel count).
%                     .block_epochs     epochs per streaming block.
%                     .parallel_blocks  run blocks on a parallel pool.
%                     .force_legacy     use the full-basis implementation.
%                     .artifact_threshold_override  fixed threshold, skip SENSAI.
%                     .verbose          print stage timings.
%
%   This is the whole-matrix interface. The computation is gedai_band_engine,
%   which GEDAI.m drives directly so that the cleaned band goes straight into
%   its accumulator; here it is collected into cleaned_data instead. Both give
%   the same values. Two implementations sit behind it - streaming for the
%   global-threshold case with many epochs, full-basis (legacy) otherwise -
%   selected as documented in gedai_band_engine.

if isempty(eeg_data)
    error('Cannot process empty data');
end
if ~ismatrix(eeg_data)
    error('Input EEG data must be a 2D matrix (channels x samples).');
end
if ~isa(eeg_data, 'double') && ~isa(eeg_data, 'single')
    eeg_data = double(eeg_data); % Only cast if not already float
end

if nargin < 9  || isempty(signal_type),  signal_type  = 'eeg'; end
if nargin < 10 || isempty(minThreshold), minThreshold = 0;     end
if nargin < 11 || isempty(maxThreshold), maxThreshold = 12;    end
if nargin < 12 || isempty(smoothing_window_seconds), smoothing_window_seconds = Inf; end
if nargin < 13, percentile_threshold = []; end
if nargin < 14 || isempty(rank_truncation), rank_truncation = true; end
if nargin < 15 || isempty(opts), opts = struct; end

src = gedai_source('matrix', eeg_data);
state = gedai_band_engine('begin', src, srate, chanlocs, artifact_threshold_type, epoch_size, ...
    refCOV, optimization_type, parallel, signal_type, minThreshold, maxThreshold, ...
    smoothing_window_seconds, percentile_threshold, rank_truncation, opts);

cleaned_data = zeros(size(eeg_data), state.cls);
if state.want_artifacts
    artifacts_data = zeros(size(eeg_data), state.cls);
else
    artifacts_data = [];
end
while ~state.done
    [state, seg, first, last, art] = gedai_band_engine('step', state);
    cleaned_data(:, first:last) = seg;
    if state.want_artifacts
        artifacts_data(:, first:last) = art;
    end
end
[SENSAI_score, artifact_threshold_out, ENOVA] = gedai_band_engine('finish', state);

if state.cast_back
    cleaned_data = double(cleaned_data);
    if ~isempty(artifacts_data), artifacts_data = double(artifacts_data); end
end
end
