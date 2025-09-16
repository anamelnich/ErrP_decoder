function plotGroupERPandPerf(params)
% plotGroupERPandPerf  Grand-average ERP (±SEM) across subjects + performance summary.
%
% Usage:
%   plotGroupERPandPerf(params)
%
% Expects each subject*.mat to contain:
%   S.bestItrData.data        [nSamples x nChans x nTrials]
%   S.bestItrData.labels      [nTrials x 1] (1 vs 0)
%   S.bestItrData.performance struct with fields: auprc, accuracy, tpr, tnr
%
% Matches your ERP preprocessing/formatting (ROI, baseline window, colors).

% ----- Configuration -----
files = {'subject2.mat','subject3.mat','subject4.mat','subject6.mat','subject7.mat'};
ROI   = {'PZ'};          % same ROI
yL    = [-10 10];         % same y-limits
shadeWindow = [0.2 0.9];  % optional shaded analysis window

% ----- Prep -----
roiIdx = find(ismember(params.chanLabels, ROI));
if isempty(roiIdx)
    error('ROI channels (%s) not found in params.chanLabels.', strjoin(ROI, ', '));
end

t  = params.epochTime(:);
nT = numel(t);

allA = []; % [nT x nSubj]
allB = [];

auprc = []; acc = []; tnr = []; tpr = [];

% ----- Load each subject and compute per-subject ERPs (same preprocessing) -----
for f = 1:numel(files)
    S = load(files{f});
    if ~isfield(S,'bestItrData')
        error('%s: missing bestItrData struct.', files{f});
    end
    C = S.bestItrData;

    req = {'data','labels','performance'};
    if ~all(isfield(C, req))
        error('%s.bestItrData missing fields: %s', files{f}, strjoin(req(~isfield(C,req)), ', '));
    end
    prreq = {'auprc','accuracy','tpr','tnr'};
    if ~all(isfield(C.performance, prreq))
        error('%s.bestItrData.performance missing fields: %s', files{f}, strjoin(prreq(~isfield(C.performance,prreq)), ', '));
    end

    data   = C.data;      % [nSamples x nChans x nTrials]
    labels = C.labels(:); % [nTrials x 1]

    % --- Baseline correction (exactly like your function) ---
    bl_idx = find(t >= params.baseline_window(1) & t <= params.baseline_window(2));
    if isempty(bl_idx)
        error('Baseline window does not overlap epochTime.');
    end
    baseline = mean(data(bl_idx, :, :), 1);
    data = data - baseline;  % implicit expansion

    % --- Average across ROI channels then across trials ---
    X_roi = squeeze(mean(data(:, roiIdx, :), 2, 'omitnan'));  % [nSamples x nTrials]

    % Conditions
    ixA = (labels == 1);
    ixB = (labels == 0);

    erpA = mean(X_roi(:, ixA), 2, 'omitnan');  % [nSamples x 1]
    erpB = mean(X_roi(:, ixB), 2, 'omitnan');

    allA(:, end+1) = erpA; %#ok<AGROW>
    allB(:, end+1) = erpB; %#ok<AGROW>

    % Performance
    auprc(end+1) = C.performance.auprc;    %#ok<AGROW>
    acc(end+1)   = C.performance.accuracy; %#ok<AGROW>
    tnr(end+1)   = C.performance.tnr;      %#ok<AGROW>
    tpr(end+1)   = C.performance.tpr;      %#ok<AGROW>
end

nSubj = size(allA, 2);

% ----- Grand means & SEM -----
mA  = mean(allA, 2, 'omitnan');  seA = std(allA, 0, 2, 'omitnan') ./ sqrt(nSubj);
mB  = mean(allB, 2, 'omitnan');  seB = std(allB, 0, 2, 'omitnan') ./ sqrt(nSubj);

% ----- Plot 1: Grand-average ERP with SEM (same style cues as your function) -----
figure('Color','w', 'Units','inches', 'Position',[1 1 4 3]);
ax = axes; hold(ax,'on');

% Shaded analysis window
patch(ax, [shadeWindow(1) shadeWindow(2) shadeWindow(2) shadeWindow(1)], ...
          [yL(1) yL(1) yL(2) yL(2)], ...
          [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

% SEM bands
shadedError(ax, t, mA, seA, params.plotColor{1}, 0.15);
shadedError(ax, t, mB, seB, params.plotColor{5}, 0.15);

% Mean lines
hA = plot(ax, t, mA, 'LineWidth',2, 'Color', params.plotColor{1});
hB = plot(ax, t, mB, 'LineWidth',2, 'Color', params.plotColor{5});

% Reference lines
xline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');
yline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');

% Axes & labels
xlim(ax, [-0.1 1]);
ylim(ax, yL);
xticks(ax, 0:0.1:max(t));
xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
legend(ax, [hA hB], {'Negative Feedback','No Feedback'}, ...
    'Box','on', 'FontSize',10, 'Location','northeast');
set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
box(ax,'off'); hold(ax,'off');

% ----- Plot 2: Classification performance (AUPRC, ACC, TPR, TNR) -----
metrics = {'AUPRC','Accuracy','TPR','TNR'};
vals = [auprc(:) acc(:) tpr(:) tnr(:)];   % [nSubj x 4]
m    = mean(vals, 1, 'omitnan');
se   = std(vals, 0, 1, 'omitnan') ./ sqrt(nSubj);

% UT-style burnt-orange palette (dark → light)
colors = [191  87   0;   % #BF5700  Burnt Orange
          217 119  50;   % #D97732
          232 140  74;   % #E88C4A
          243 167 108] ./ 255; % #F3A76C

figure('Color','w', 'Units','inches', 'Position',[1 1 5.4 3.2]);
ax2 = axes('Parent', gcf); hold(ax2,'on');

% Bars with distinct shades
barWidth = 0.6;
for k = 1:4
    bar(k, m(k), barWidth, 'FaceColor', colors(k,:), 'EdgeColor','none');
end

% Error bars (SEM)
errorbar(ax2, 1:4, m, se, 'k', 'LineStyle','none', 'LineWidth',1.2, 'CapSize',8);

% Subject dots (jittered, semi-transparent)
for k = 1:4
    xj = k + 0.06*randn(nSubj,1);
    scatter(ax2, xj, vals(:,k), 18, 'filled', ...
        'MarkerFaceColor',[0.25 0.25 0.25], 'MarkerFaceAlpha',0.7, ...
        'MarkerEdgeColor','none');
end

% Mean value labels above bars
for k = 1:4
    text(k, m(k) + 0.07, sprintf('%.2f', m(k)), ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
        'FontName','Arial', 'FontSize',9, 'FontWeight','bold');
end

% Axes & styling
xlim(ax2, [0.4 4.6]);
ylim(ax2, [0.45 0.8]);           % per your spec
xticks(ax2, 1:4); xticklabels(ax2, metrics);
ylabel(ax2,'Score', 'FontName','Arial', 'FontSize',10);
set(ax2, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
box(ax2,'off');

% Optional: subtle y-grid for readability (publication-friendly)
ax2.YGrid = 'on';
ax2.GridAlpha = 0.15;
ax2.GridColor = [0 0 0];


hold(ax2,'off');


end

% ---------- Helper (local) ----------
function shadedError(ax, t, m, se, colorSpec, alphaVal)
% Draw mean ± SEM band behind a line
lo = m - se;
hi = m + se;
patch(ax, [t; flipud(t)]', [lo; flipud(hi)]', colorSpec, ...
    'FaceAlpha', alphaVal, 'EdgeColor','none', 'HandleVisibility','off');
end
