function plotRunPruning(origData, bestData)
% plotRunPruning  Publication-quality plot of trials removed per run
%
%   plotRunPruning(origData, bestData)
%   Computes and plots the percentage of trials removed per run (file_id) for:
%     A) Combined trials
%     B) Distractor trials only
%     C) No-distractor trials only
%   Displays as a 3×1 tiledlayout with consistent styling for high-impact journals.

% Extract unique run IDs
fileIDs = unique(origData.file_id(:))';
nRuns = numel(fileIDs);

% Initialize removal percentages
pctRemoved  = zeros(1,nRuns);
pctRemovedD = zeros(1,nRuns);
pctRemovedN = zeros(1,nRuns);
for i = 1:nRuns
    id = fileIDs(i);
    oMask = origData.file_id == id;
    bMask = bestData.file_id == id;
    % Combined
    pctRemoved(i)  = (sum(oMask) - sum(bMask)) / sum(oMask) * 100;
    % Distractor only
    dOm = oMask & origData.labels==1;
    dBm = bMask & bestData.labels==1;
    if sum(dOm)>0
        pctRemovedD(i) = (sum(dOm) - sum(dBm)) / sum(dOm) * 100;
    end
    % No-distractor only
    nOm = oMask & origData.labels==0;
    nBm = bMask & bestData.labels==0;
    if sum(nOm)>0
        pctRemovedN(i) = (sum(nOm) - sum(nBm)) / sum(nOm) * 100;
    end
end

% Qualitative color palette
colors = [0.2 0.4 0.6;  % Combined (blue)
          0.8 0.3 0.3;  % Distractor (red)
          0.3 0.6 0.3]; % No-distractor (green)

% Plot setup
figure('Color','w','Units','inches','Position',[2 2 5 8]);
t = tiledlayout(3,1, 'Padding','compact', 'TileSpacing','compact');
dataMat     = [pctRemoved; pctRemovedD; pctRemovedN];
ylabels     = {'Combined','Distractor','No distractor'};
panelLabels = {'A','B','C'};

for p = 1:3
    ax = nexttile;
    hold(ax,'on');
    % Bar chart
    bar(ax, fileIDs, dataMat(p,:), 'FaceColor', colors(p,:), 'BarWidth',0.7);
    % Grid styling
    yMax = ceil(max(dataMat(p,:))/10)*10;
    ylim(ax,[0 100]);
    yticks(ax, linspace(0, 100, 5));
    grid(ax,'on');
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.5;
    % Labels and aesthetics
    xlabel(ax, 'Run ID',   'FontName','Arial', 'FontSize',10);
    ylabel(ax, '% Removed', 'FontName','Arial', 'FontSize',10);
    title(ax, sprintf('%s   %s', panelLabels{p}, ylabels{p}), ...
        'FontName','Arial', 'FontSize',12, 'FontWeight','bold');
    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    hold(ax,'off');
end


end

