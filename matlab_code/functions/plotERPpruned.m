function plotERPpruned(origData, bestData, params)
% plotERPpruned  Publication-quality grand-average diff waves with RT markers
%
%   plotERPpruned(origData, bestData, params)
%   Creates a two-panel figure (before & after pruning) with enhanced
%   styling and mean Reaction Time (RT) vertical lines per condition.

% Define electrode ROIs
LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};
lIdx = find( ismember(params.chanLabels, LeftElec) );
rIdx = find( ismember(params.chanLabels, RightElec) );

% Prepare figure
figure('Color','w', 'Units','inches', 'Position',[1 1 4 6]);
T = tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact');
annotations = {'A: Before pruning', 'B: After pruning'};
datasets = {origData, bestData};
yL = [-3 3]; % consistent y-limits

for p = 1:2
    ax = nexttile;
    hold(ax,'on');
    D = datasets{p};

    % trial indices
    dTrials  = D.labels==1;
    ndTrials = D.labels==0;

    % compute grand-averages
    avgDl = squeeze(mean(mean(D.data(:,lIdx,dTrials),2),3));
    avgDr = squeeze(mean(mean(D.data(:,rIdx,dTrials),2),3));
    diffD  = avgDr - avgDl;
    avgNdl = squeeze(mean(mean(D.data(:,lIdx,ndTrials),2),3));
    avgNdr = squeeze(mean(mean(D.data(:,rIdx,ndTrials),2),3));
    diffND = avgNdr - avgNdl;

    % gray shading
    patch([0.2 0.5 0.5 0.2], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

    % plot waveforms
    h1 = plot(ax, params.epochTime, diffD,  'LineWidth',2, 'Color', params.plotColor{1});
    h2 = plot(ax, params.epochTime, diffND, 'LineWidth',2, 'Color', params.plotColor{5});

    % mean RT lines
    meanRT_d  = mean(D.RT(dTrials))/1000;
    meanRT_nd = mean(D.RT(ndTrials))/1000;
    meanRT_diff = (meanRT_nd - meanRT_d)*1000;
    h3 = xline(ax, meanRT_d,  '--', 'LineWidth',1.5, 'Color', params.plotColor{1},'HandleVisibility','off');
    h4 = xline(ax, meanRT_nd, '--', 'LineWidth',1.5, 'Color', params.plotColor{5},'HandleVisibility','off');

    % zero reference lines (excluded from legend)
    xline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');
    yline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');

    % axes limits and ticks
    xlim(ax,[-0.1 0.65]);
    ylim(ax,yL);
    xticks(ax,0:0.1:max(params.epochTime));

    % labels and title
    xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
    ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
    titleStr = sprintf('%s — nd-d: %.0f ms', annotations{p}, meanRT_diff);
    title(ax, titleStr, 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % legend 
    lg = legend(ax, [h1 h2], {
        'Distractor', 'No distractor'
    }, 'Box','on', 'FontSize',10, 'Location','northeast');

    % panel label (A/B)
    text(ax, -0.08, 1.02, annotations{p}(1), ...
        'Units','normalized', 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % aesthetic tweaks
    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    box(ax,'off');
    hold(ax,'off');
end

end

