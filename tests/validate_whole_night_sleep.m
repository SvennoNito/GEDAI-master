function metrics = validate_whole_night_sleep(inputSet, stageSet, wholeNightSet, scoringFile, outputDir)
%VALIDATE_WHOLE_NIGHT_SLEEP Compare whole-night GEDAI with a stage-wise baseline.
%
% The event locations are selected only from the stage-specific output, then
% their amplitude is measured at exactly the same samples in the input and
% whole-night files. This makes a lower whole-night/stage-specific ratio an
% interpretable attenuation result rather than a changed detector count.
%
% The three measures are deliberately conservative, single-channel proxies:
%   N3 slow waves: 0.3-4 Hz negative peaks at Fz.
%   N2 K-complex candidates: large 0.3-4 Hz negative peaks at Fz.
%   N2 spindles: 11-16 Hz envelope bursts at Fz.
%
% It reads one channel through memory mapping, so it is suitable for the
% 8+ GB DROP recordings without loading the full recordings into RAM.

arguments
    inputSet     (1,:) char
    stageSet     (1,:) char
    wholeNightSet (1,:) char
    scoringFile  (1,:) char
    outputDir    (1,:) char = fullfile(tempdir, 'gedai-whole-night-validation')
end

if ~isfolder(outputDir), mkdir(outputDir); end

[raw, hdrRaw]   = local_read_fz(inputSet);
[stage, hdrStage] = local_read_fz(stageSet);
[whole, hdrWhole] = local_read_fz(wholeNightSet);

if hdrRaw.srate ~= hdrStage.srate || hdrRaw.srate ~= hdrWhole.srate
    error('All recordings must have the same sampling rate.');
end
fs = hdrRaw.srate;
n  = min([numel(raw), numel(stage), numel(whole)]);
raw = double(raw(1:n)); stage = double(stage(1:n)); whole = double(whole(1:n));

% DROP scoring uses the standard 0/1/2/3/5 codes (wake/N1/N2/N3/REM).
scoring = readmatrix(scoringFile);
scoring = scoring(:, 1)';
epochSamples = 30 * fs;
stageAtSample = repelem(scoring, epochSamples);
stageAtSample = stageAtSample(1:min(n, numel(stageAtSample)));
n = numel(stageAtSample);
raw = raw(1:n); stage = stage(1:n); whole = whole(1:n);

slowStage = local_bandpass(stage, fs, [0.3 4]);
slowRaw   = local_bandpass(raw,   fs, [0.3 4]);
slowWhole = local_bandpass(whole, fs, [0.3 4]);

% A 1.5-s window covers the negative and following positive lobe. The
% amplitude bounds deliberately rule out small fluctuations and gross motion.
slowEvents = local_slow_events(slowStage, stageAtSample == 3, fs, 75, 500);
kcEvents   = local_slow_events(slowStage, stageAtSample == 2, fs, 100, 500);

slowMetrics = local_ptp_metrics('N3 slow waves', slowEvents, slowRaw, slowStage, slowWhole);
kcMetrics   = local_ptp_metrics('N2 K-complex candidates', kcEvents, slowRaw, slowStage, slowWhole);

sigmaStage = local_bandpass(stage, fs, [11 16]);
sigmaRaw   = local_bandpass(raw,   fs, [11 16]);
sigmaWhole = local_bandpass(whole, fs, [11 16]);
spindleEvents = local_spindle_events(sigmaStage, stageAtSample == 2, fs);
spindleMetrics = local_spindle_metrics(spindleEvents, sigmaRaw, sigmaStage, sigmaWhole);

metrics = [slowMetrics; kcMetrics; spindleMetrics];
writetable(metrics, fullfile(outputDir, 'whole_night_signal_retention.csv'));
save(fullfile(outputDir, 'whole_night_signal_retention.mat'), 'metrics', 'slowEvents', 'kcEvents', 'spindleEvents');

local_plot_examples(outputDir, fs, raw, stage, whole, slowEvents, kcEvents, spindleEvents);
fprintf('\nFz whole-night validation (whole-night / stage-specific):\n');
disp(metrics)
end

function [x, hdr] = local_read_fz(setFile)
hdr = load(setFile, '-mat');
if ~isfield(hdr, 'data') || ~ischar(hdr.data)
    error('%s is not an EEGLAB two-file .set/.fdt dataset.', setFile);
end
fz = find(strcmpi({hdr.chanlocs.labels}, 'E21'), 1); % DROP EGI-256 Fz
if isempty(fz)
    error('Fz (EGI E21) was removed from %s; select a retained frontal channel.', setFile);
end
fdtFile = fullfile(fileparts(setFile), hdr.data);
if ~isfile(fdtFile), error('Cannot find FDT file: %s', fdtFile); end
m = memmapfile(fdtFile, 'Format', {'single', [hdr.nbchan hdr.pnts], 'x'}, 'Writable', false);
x = m.Data.x(fz, :);
hdr = rmfield(hdr, setdiff(fieldnames(hdr), {'srate','pnts','nbchan'}));
end

function y = local_bandpass(x, fs, rangeHz)
[b, a] = butter(4, rangeHz / (fs / 2), 'bandpass');
y = filtfilt(b, a, x(:)')';
end

function events = local_slow_events(x, mask, fs, lowerAmp, upperAmp)
[~, loc] = findpeaks(-x, 'MinPeakDistance', round(0.8 * fs), ...
    'MinPeakProminence', 30);
halfBefore = round(0.5 * fs); halfAfter = round(1.0 * fs);
events = zeros(0, 2);
for i = 1:numel(loc)
    s = loc(i) - halfBefore; e = loc(i) + halfAfter;
    if s < 1 || e > numel(x) || ~all(mask(s:e)), continue; end
    a = peak2peak(x(s:e));
    if a >= lowerAmp && a <= upperAmp
        events(end+1, :) = [s e]; %#ok<AGROW>
    end
end
end

function events = local_spindle_events(x, mask, fs)
env = movmean(abs(hilbert(x)), round(0.15 * fs));
threshold = prctile(env(mask), 95);
above = env >= threshold & mask;
edge = diff([false; above(:); false]);
starts = find(edge == 1); ends = find(edge == -1) - 1;
events = zeros(0, 2);
for i = 1:numel(starts)
    duration = (ends(i) - starts(i) + 1) / fs;
    if duration >= 0.5 && duration <= 2.5
        events(end+1, :) = [starts(i), ends(i)]; %#ok<AGROW>
    end
end
end

function row = local_ptp_metrics(label, events, raw, stage, whole)
if isempty(events)
    row = local_empty_row(label);
    return
end
stageAmp = arrayfun(@(i) peak2peak(stage(events(i,1):events(i,2))), 1:size(events,1))';
rawAmp   = arrayfun(@(i) peak2peak(raw(events(i,1):events(i,2))),   1:size(events,1))';
wholeAmp = arrayfun(@(i) peak2peak(whole(events(i,1):events(i,2))), 1:size(events,1))';
row = local_row(label, stageAmp, rawAmp, wholeAmp);
end

function row = local_spindle_metrics(events, raw, stage, whole)
if isempty(events)
    row = local_empty_row('N2 spindles');
    return
end
stageAmp = arrayfun(@(i) max(abs(hilbert(stage(events(i,1):events(i,2))))), 1:size(events,1))';
rawAmp   = arrayfun(@(i) max(abs(hilbert(raw(events(i,1):events(i,2))))),   1:size(events,1))';
wholeAmp = arrayfun(@(i) max(abs(hilbert(whole(events(i,1):events(i,2))))), 1:size(events,1))';
row = local_row('N2 spindles', stageAmp, rawAmp, wholeAmp);
end

function row = local_row(label, stageAmp, rawAmp, wholeAmp)
ratio = wholeAmp ./ stageAmp;
row = table(string(label), numel(stageAmp), median(rawAmp), median(stageAmp), median(wholeAmp), ...
    median(ratio), prctile(ratio, 10), prctile(ratio, 90), ...
    'VariableNames', {'signal','nEvents','rawMedian_uV','stageSpecificMedian_uV', ...
    'wholeNightMedian_uV','medianRetention','p10Retention','p90Retention'});
end

function row = local_empty_row(label)
row = table(string(label), 0, NaN, NaN, NaN, NaN, NaN, NaN, ...
    'VariableNames', {'signal','nEvents','rawMedian_uV','stageSpecificMedian_uV', ...
    'wholeNightMedian_uV','medianRetention','p10Retention','p90Retention'});
end

function local_plot_examples(outputDir, fs, raw, stage, whole, slowEvents, kcEvents, spindleEvents)
groups = {slowEvents, kcEvents, spindleEvents};
labels = {'N3 slow wave', 'N2 K-complex candidate', 'N2 spindle'};
figure('Color', 'w', 'Position', [100 100 1300 780]);
for r = 1:3
    nexttile = subplot(3, 1, r); %#ok<NASGU>
    events = groups{r};
    if isempty(events), title([labels{r} ': no candidate']); continue; end
    i = ceil(size(events, 1) / 2); % robust representative, not a cherry-picked maximum
    s = events(i,1); e = events(i,2); t = ((s:e) - s) / fs;
    plot(t, raw(s:e), 'Color', [.65 .65 .65]); hold on
    plot(t, stage(s:e), 'k', 'LineWidth', 1.1);
    plot(t, whole(s:e), 'Color', [0 .35 .8], 'LineWidth', 1.1);
    grid on; ylabel('\muV'); title(labels{r});
    if r == 1, legend({'input','stage-specific','whole-night'}, 'Location', 'best'); end
    if r == 3, xlabel('seconds from event-window start'); end
end
exportgraphics(gcf, fullfile(outputDir, 'whole_night_representative_events.png'), 'Resolution', 160);
close(gcf)
end
