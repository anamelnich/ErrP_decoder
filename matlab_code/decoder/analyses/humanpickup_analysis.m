filename = './../../data/subject7/s7_TAMERlabels.xlsx';
tamer_labels = readmatrix(filename);
%%
tamer_labels = reshape(tamer_labels, [], 1);
tamer_labels_calibration = tamer_labels(1:800);
tamer_labels_online = tamer_labels(801:end);
%% for subject 6, remove label 601

epochs.data(:,:,601) = [];
epochs.labels(601) = [];
epochs.file_id(601)=[];

%% ERP
human_mask = tamer_labels_calibration(:,1) == 1;
% human_mask = tamer_labels_online(1:602,1) == 1;
neg_mask = epochs.labels == 1;
neutral_mask = (tamer_labels_calibration == 0) & (epochs.labels == 0); 

sum_h = sum(human_mask);
sum_n = sum(neg_mask);
sum_neu = sum(neutral_mask);

conflict = human_mask & neg_mask;
fprintf('Counts: human=%d, neg=%d, neutral=%d, total=%d\n', ...
        sum_h, sum_n, sum_neu, sum_h+sum_n+sum_neu);
fprintf('Conflicts (human & neg) = %d\n', sum(conflict));

%%
assert(numel(human_mask)==size(epochs.data,3), 'human_mask length mismatch');

D = struct('data', epochs.data, 'labels', epochs.labels);
cfg.plotColor = { [0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19] }; 


% Plot
masks  = {human_mask, neg_mask, neutral_mask};
names  = {'Human Pickup', 'Negative feedback', 'Neutral'};
plotThreeERPs(D, cfg, masks, names);
%% Posteriors
[posterior, epoch] = singleClassification(decoder, epochs.data);

%% Class subsets
classH = posterior(human_mask);
classN = posterior(neg_mask);
classNeu = posterior(neutral_mask);

% Setup figure
figure('Color','w','Units','inches','Position',[1 1 6 4]); hold on;

% Plot histograms (normalized so areas = 1)
histogram(classH, 'Normalization','probability', ...
    'FaceColor',[0.2 0.6 0.8], 'FaceAlpha',0.5, 'EdgeColor','none', 'BinWidth',0.05);
histogram(classN, 'Normalization','probability', ...
    'FaceColor',[0.9 0.4 0.2], 'FaceAlpha',0.5, 'EdgeColor','none','BinWidth',0.05);
histogram(classNeu, 'Normalization','probability', ...
    'FaceColor',[0.4 0.7 0.3], 'FaceAlpha',0.5, 'EdgeColor','none','BinWidth',0.05);

% Threshold line
xline(decoder.threshold, '--k', 'LineWidth',2);

% Labels and legend
xlabel('Posterior probability','FontName','Arial','FontSize',12);
ylabel('Probability','FontName','Arial','FontSize',12);
legend({'Human','Negative','Neutral','Threshold'}, 'Location','best');
title('Posterior distributions across classes','FontName','Arial','FontSize',12);

set(gca,'FontName','Arial','FontSize',10,'LineWidth',1);
hold off;

%%
% Predicted class 0 (below threshold)
pred_class0 = posterior < decoder.threshold;

% Count per trial type
nHuman  = sum(pred_class0 & human_mask);
nNeg    = sum(pred_class0 & neg_mask);
nNeutral= sum(pred_class0 & neutral_mask);

% Total counts for percentage
totHuman   = sum(human_mask);
totNeg     = sum(neg_mask);
totNeutral = sum(neutral_mask);

% Percent classified as class 0
pctHuman   = 100 * sum(pred_class0 & human_mask)   / totHuman;
pctNeg     = 100 * sum(pred_class0 & neg_mask)     / totNeg;
pctNeutral = 100 * sum(pred_class0 & neutral_mask) / totNeutral;

C = {[0.2 0.6 0.8], [0.9 0.4 0.2], [0.4 0.7 0.3]};

% --- Plot ---
figure('Color','w','Units','inches','Position',[1 1 5 3.5]); hold on;

vals = [pctHuman; pctNeg; pctNeutral];   % column vector
b = bar(vals, 'FaceColor','flat');

% Assign colors
for i = 1:3
    b.CData(i,:) = C{i};
end

% Set x-axis labels properly
set(gca,'XTick',1:3, 'XTickLabel',{'Human','Negative','Neutral'});

ylabel('% classified as class 0');
ylim([0 100]); % y-axis in percent
title('Percentage of trials classified as class 0');

% Annotate percentages on top
xtips = 1:3;
ytips = vals;
labels = arrayfun(@(x) sprintf('%.1f%%',x), ytips, 'UniformOutput',false);
text(xtips, ytips, labels, 'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', 'FontSize',10);

hold off;

%% ================= Helper Functions ================= %%

function h = plotThreeERPs(D, params, masks, names)
% plotThreeERPs  Single-panel ERP plot of three conditions (masks)
%
%   h = plotThreeERPs(D, params, masks, names)
%
% Inputs
%   D.data   : [nSamples x nChans x nTrials]
%   D.labels : [nTrials x 1]  (not required for plotting; you can still pass it)
%   params.chanLabels   : cellstr of channel names
%   params.epochTime    : [nSamples x 1] time (s)
%   params.baseline_window : [t0 t1] seconds (e.g., [-0.2 0])
%   params.plotColor    : cell array of colors; will fallback if <3 provided
%
%   masks : 1x3 logical vectors over trials (e.g., {human_mask, neg_mask, neutral_mask})
%   names : 1x3 cellstr legend names
%
% Output
%   h : struct with plot handles
%
% Example ROI uses FCZ; change ROI below as needed.

% ----- ROI -----
ROI = {'FCZ'};
roiIdx = find(ismember(params.chanLabels, ROI));
if isempty(roiIdx)
    error('ROI channels not found in params.chanLabels.');
end

% ----- Baseline correction -----
t = params.epochTime(:);
bl = params.baseline_window;
bl_idx = find(t >= bl(1) & t <= bl(2));
if isempty(bl_idx), error('baseline_window does not overlap epochTime'); end
baseline = mean(D.data(bl_idx, :, :), 1, 'omitnan');
X = D.data - baseline; % same size as D.data

% ----- Reduce to ROI: [nSamples x nTrials] -----
X_roi = squeeze(mean(X(:, roiIdx, :), 2, 'omitnan'));

% ----- Colors -----
fallbackColors = {[0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]}; % MATLAB-ish palette
C = fallbackColors;
if isfield(params, 'plotColor') && numel(params.plotColor) >= 3
    C = params.plotColor(1:3);
end

% ----- Figure -----
figure('Color','w', 'Units','inches', 'Position',[1 1 5 3.5]);
ax = axes; hold(ax,'on');

% Optional analysis window patch (adjust or remove)
yL = [-10 10];
if isfield(params, 'analysis_window') && numel(params.analysis_window)==2
    aw = params.analysis_window;
    patch([aw(1) aw(2) aw(2) aw(1)], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.4, 'HandleVisibility','off');
end

% ----- Compute means and plot -----
h.lines = gobjects(1,3);
for k = 1:3
    mk = masks{k};
    if numel(mk) ~= size(X_roi,2)
        error('Mask %d length (%d) does not match nTrials (%d).', k, numel(mk), size(X_roi,2));
    end
    erp_k = mean(X_roi(:, mk), 2, 'omitnan');
    h.lines(k) = plot(t, erp_k, 'LineWidth', 2, 'Color', C{k});
end

% ----- Cosmetics -----
xline(ax, 0, '--', 'LineWidth',1.2, 'HandleVisibility','off');
yline(ax, 0, '--', 'LineWidth',1.2, 'HandleVisibility','off');
xlim(ax, [-0.1 1]);
ylim(ax, yL);
xlabel(ax, 'Time (s)', 'FontName','Arial', 'FontSize',10);
ylabel(ax, 'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
legend(ax, h.lines, names, 'Location','northeast', 'Box','on', 'FontSize',10);
set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1); box(ax,'off'); grid(ax,'off');
hold(ax,'off');

h.ax = ax;
end
