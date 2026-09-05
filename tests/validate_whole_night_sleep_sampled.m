function metrics = validate_whole_night_sleep_sampled(inputSet, stageSet, wholeNightSet, scoringFile, outputDir, maxEpochsPerStage)
%VALIDATE_WHOLE_NIGHT_SLEEP_SAMPLED Fast, stratified event-retention test.
%
% This is the network-friendly counterpart of validate_whole_night_sleep.
% It takes evenly spaced N3 and N2 epochs, identifies events in the
% stage-specific baseline, then measures the same sample windows in the raw
% and whole-night recordings. It never loads a complete FDT file.

arguments
    inputSet (1,:) char
    stageSet (1,:) char
    wholeNightSet (1,:) char
    scoringFile (1,:) char
    outputDir (1,:) char
    maxEpochsPerStage (1,1) double {mustBePositive, mustBeInteger} = 24
end
if ~isfolder(outputDir), mkdir(outputDir); end

raw = local_open(inputSet); stage = local_open(stageSet); whole = local_open(wholeNightSet);
assert(raw.fs == stage.fs && raw.fs == whole.fs, 'Sampling rates differ.');
fs = raw.fs; epN = 30 * fs;
scores = readmatrix(scoringFile); scores = scores(:,1)';
nEpoch = min([floor(raw.pnts/epN), floor(stage.pnts/epN), floor(whole.pnts/epN), numel(scores)]);

n3 = local_even_epochs(find(scores(1:nEpoch) == 3), maxEpochsPerStage);
n2 = local_even_epochs(find(scores(1:nEpoch) == 2), maxEpochsPerStage);

[slowStage, slowRaw, slowWhole, slowExample] = local_slow(n3, raw, stage, whole, fs, 75);
[kcStage, kcRaw, kcWhole, kcExample] = local_slow(n2, raw, stage, whole, fs, 100);
[spindleStage, spindleRaw, spindleWhole, spindleExample] = local_spindles(n2, raw, stage, whole, fs);

metrics = [local_row(sprintf('N3 slow waves (%d stratified epochs)', numel(n3)), slowStage, slowRaw, slowWhole); ...
           local_row(sprintf('N2 K-complex candidates (%d stratified epochs)', numel(n2)), kcStage, kcRaw, kcWhole); ...
           local_row(sprintf('N2 spindles (%d stratified epochs)', numel(n2)), spindleStage, spindleRaw, spindleWhole)];
writetable(metrics, fullfile(outputDir, 'whole_night_signal_retention_sampled.csv'));
save(fullfile(outputDir, 'whole_night_signal_retention_sampled.mat'), 'metrics', 'n2', 'n3');
local_plot(outputDir, fs, slowExample, kcExample, spindleExample);
fprintf('\nSampled Fz validation (whole-night / stage-specific):\n'); disp(metrics)
end

function reader = local_open(setFile)
h = load(setFile, '-mat');
fz = find(strcmpi({h.chanlocs.labels}, 'E21'), 1);
assert(~isempty(fz), 'Fz (E21) is absent: %s', setFile);
fdt = fullfile(fileparts(setFile), h.data);
% Construct field-by-field: memmapfile is itself an object with a struct
% method, so passing it directly to struct(...) invokes that method.
reader = struct();
reader.map = memmapfile(fdt, 'Format', {'single',[h.nbchan h.pnts],'x'}, 'Writable', false);
reader.fz = fz; reader.pnts = h.pnts; reader.fs = h.srate;
end

function idx = local_even_epochs(idx, n)
assert(~isempty(idx), 'Requested sleep stage is absent from scoring.');
idx = idx(round(linspace(1, numel(idx), min(n, numel(idx)))));
end

function [stageAmp, rawAmp, wholeAmp, example] = local_slow(epochs, raw, stage, whole, fs, minAmp)
stageAmp = []; rawAmp = []; wholeAmp = []; example = [];
for ep = epochs
    [xR, xS, xW] = local_epoch(ep, raw, stage, whole, fs);
    sR = local_band(xR, fs, [.3 4]); sS = local_band(xS, fs, [.3 4]); sW = local_band(xW, fs, [.3 4]);
    [~, loc] = findpeaks(-sS, 'MinPeakDistance', round(.8*fs), 'MinPeakProminence', 30);
    for p = loc(:)'
        a = p - round(.5*fs); b = p + round(1*fs);
        if a <= 2*fs || b > numel(sS)-2*fs, continue; end
        baseline = peak2peak(sS(a:b));
        if baseline < minAmp || baseline > 500, continue; end
        stageAmp(end+1,1) = baseline; %#ok<AGROW>
        rawAmp(end+1,1) = peak2peak(sR(a:b)); %#ok<AGROW>
        wholeAmp(end+1,1) = peak2peak(sW(a:b)); %#ok<AGROW>
        if isempty(example), example = [xR(a:b), xS(a:b), xW(a:b)]; end
    end
end
end

function [stageAmp, rawAmp, wholeAmp, example] = local_spindles(epochs, raw, stage, whole, fs)
stageAmp = []; rawAmp = []; wholeAmp = []; example = [];
for ep = epochs
    [xR, xS, xW] = local_epoch(ep, raw, stage, whole, fs);
    sR = local_band(xR, fs, [11 16]); sS = local_band(xS, fs, [11 16]); sW = local_band(xW, fs, [11 16]);
    env = movmean(abs(hilbert(sS)), round(.15*fs)); above = env >= prctile(env(2*fs:end-2*fs), 95);
    edge = diff([false; above(:); false]); starts = find(edge == 1); ends = find(edge == -1)-1;
    for i = 1:numel(starts)
        a = starts(i); b = ends(i); duration = (b-a+1)/fs;
        if a <= 2*fs || b > numel(sS)-2*fs || duration < .5 || duration > 2.5, continue; end
        stageAmp(end+1,1) = max(abs(hilbert(sS(a:b)))); %#ok<AGROW>
        rawAmp(end+1,1) = max(abs(hilbert(sR(a:b)))); %#ok<AGROW>
        wholeAmp(end+1,1) = max(abs(hilbert(sW(a:b)))); %#ok<AGROW>
        if isempty(example), example = [xR(a:b), xS(a:b), xW(a:b)]; end
    end
end
end

function [xR, xS, xW] = local_epoch(ep, raw, stage, whole, fs)
idx = (ep-1)*30*fs + (1:30*fs);
xR = double(raw.map.Data.x(raw.fz, idx))';
xS = double(stage.map.Data.x(stage.fz, idx))';
xW = double(whole.map.Data.x(whole.fz, idx))';
end

function y = local_band(x, fs, range)
[b,a] = butter(4, range/(fs/2), 'bandpass'); y = filtfilt(b,a,x);
end

function row = local_row(label, stageAmp, rawAmp, wholeAmp)
if isempty(stageAmp)
    row = table(string(label), 0, NaN, NaN, NaN, NaN, NaN, ...
        'VariableNames', {'signal','nEvents','rawMedian_uV','stageSpecificMedian_uV','wholeNightMedian_uV','medianRetention','p10Retention'});
    return
end
ratio = wholeAmp ./ stageAmp;
row = table(string(label), numel(stageAmp), median(rawAmp), median(stageAmp), median(wholeAmp), median(ratio), prctile(ratio,10), ...
    'VariableNames', {'signal','nEvents','rawMedian_uV','stageSpecificMedian_uV','wholeNightMedian_uV','medianRetention','p10Retention'});
end

function local_plot(outputDir, fs, slow, kc, spindle)
examples = {slow,kc,spindle}; labels = {'N3 slow wave','N2 K-complex candidate','N2 spindle'};
figure('Color','w','Position',[100 100 1200 700]);
for i = 1:3
    subplot(3,1,i); x = examples{i};
    if isempty(x), title([labels{i} ': no candidate']); continue; end
    t = (0:size(x,1)-1)'/fs; plot(t,x(:,1),'Color',[.65 .65 .65]); hold on
    plot(t,x(:,2),'k','LineWidth',1.1); plot(t,x(:,3),'Color',[0 .35 .8],'LineWidth',1.1);
    grid on; ylabel('\muV'); title(labels{i});
    if i==1, legend({'input','stage-specific','whole-night'},'Location','best'); end
    if i==3, xlabel('seconds'); end
end
exportgraphics(gcf, fullfile(outputDir,'whole_night_representative_events_sampled.png'),'Resolution',160); close(gcf)
end
