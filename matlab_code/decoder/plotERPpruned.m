function plotERPpruned(origData, bestData, params)
% plotERPpruned  Publication-quality grand-average ERPs over a chosen ROI
%                (no difference wave).
%
%   plotERPpruned(origData, bestData, params)
%   Creates a two-panel figure (before & after pruning) with enhanced
%   styling and mean ERP per condition (labels: 1 vs 0) averaged over the ROI.
%
% Expects:
%   D.data   -> [nSamples x nChans x nTrials]
%   D.labels -> [nTrials x 1] (1 = condition A, 0 = condition B)
%   params.chanLabels -> cellstr of channel names
%   params.epochTime  -> time vector (seconds) matching nSamples
%   params.plotColor  -> cellstr/cell of colors, e.g., params.plotColor{1}, {5}

% ----- Define electrode ROI (single set) -----
% ROI = {'FZ','FCZ','CZ','CPZ','PZ','FC1','FC2','C1','C2'};
ROI = {'FCZ'};
roiIdx = find( ismember(params.chanLabels, ROI) );
baseline_window = params.baseline_window;
baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
baselineO = mean(origData.data(baseline_idx, :, :), 1);
origData.data = origData.data - baselineO;

baselineB = mean(bestData.data(baseline_idx, :, :), 1);
bestData.data = bestData.data - baselineB;

% ----- Figure setup -----
figure('Color','w', 'Units','inches', 'Position',[1 1 4 6]);
T = tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact'); %#ok<NASGU>
annotations = {'A: Before pruning', 'B: After pruning'};
datasets   = {origData, bestData};
yL = [-20 20];  % consistent y-limits across panels

for p = 1:2
    ax = nexttile; hold(ax,'on');
    D = datasets{p};

    % Trial groups
    ixA = (D.labels == 1);   % e.g., "Distractor" / class 1
    ixB = (D.labels == 0);   % e.g., "No distractor" / class 0

    % --- Average across ROI channels then across trials ---
    % X_roi: [nSamples x nTrials]
    X_roi = squeeze(mean(D.data(:, roiIdx, :), 2, 'omitnan'));

    % Mean ERP per condition
    erpA = mean(X_roi(:, ixA), 2, 'omitnan');
    erpB = mean(X_roi(:, ixB), 2, 'omitnan');

    % --- Optional shaded analysis window (keep as in your original) ---
    patch([0.2 0.9 0.9 0.2], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

    % --- Plot ERPs ---
    hA = plot(ax, params.epochTime, erpA, 'LineWidth',2, 'Color', params.plotColor{1});
    hB = plot(ax, params.epochTime, erpB, 'LineWidth',2, 'Color', params.plotColor{5});

    % Zero reference lines (hidden from legend)
    xline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');
    yline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');

    % Axes limits & ticks
    xlim(ax, [-0.1 1]);
    ylim(ax, yL);
    xticks(ax, 0:0.1:max(params.epochTime));

    % Labels
    xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
    ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);

    % Legend (rename if you prefer other condition names)
    lg = legend(ax, [hA hB], {'Negative Feedback','No Feedback'}, ...
        'Box','on', 'FontSize',10, 'Location','northeast'); %#ok<NASGU>

    % Panel label (A/B)
    text(ax, -0.08, 1.02, annotations{p}(1), ...
        'Units','normalized', 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % Aesthetics
    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    box(ax,'off'); hold(ax,'off');
end

end


