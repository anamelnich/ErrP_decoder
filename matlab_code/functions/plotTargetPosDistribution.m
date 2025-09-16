function plotTargetPosDistribution(origData, bestData, params)
% plotTargetPosDistribution  Enhanced percent bar plots by target position
%
%   plotTargetPosDistribution(origData, bestData, params)
%
%   Inputs:
%     origData & bestData – structs with fields:
%         .tpos   [nTrials×1] target positions (1=Up,2=Right,3=Down,4=Left)
%         .labels [nTrials×1] distractor flags (1=distractor,0=no-distractor)
%     params – struct with optional fields:
%         .posLabels {4×1} cell of names, default {'Up','Right','Down','Left'}
%         .colors    2×2 cell of color specs: {beforeDistractor, beforeNo, afterDistractor, afterNo}
%
%   Creates a 2×1 tiled grouped bar plot: panel 1 = distractor, panel 2 = no-distractor.

% Default labels
if isfield(params,'posLabels'), posLabels = params.posLabels; else posLabels = {'Up','Right','Down','Left'}; end
if isfield(params,'colors'), cols = params.colors; else cols = {{[0.2 0.6 0.8],[0.7 0.7 0.7]}, {[0.3 0.8 0.4],[0.8 0.6 0.3]}}; end

% Prepare figure
figure('Color','w','Units','inches','Position',[1 1 6 4]);
T = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
conditions = [1,0]; condNames = {'Distractor','No distractor'};
dataSets   = {origData,bestData}; dsNames = {'Before','After'};

for c=1:2
    ax = nexttile;
    % compute percent for each dataset and position
    pct = zeros(2,4);
    for d=1:2
        D = dataSets{d};
        mask = D.labels==conditions(c);
        for k=1:4
            pct(d,k) = sum(mask & D.tpos==k)/sum(mask)*100;
        end
    end
    % grouped bar
    b = bar(ax, pct','grouped','BarWidth',0.7);
    % set colors
    b(1).FaceColor = cols{1}{c};
    b(2).FaceColor = cols{2}{c};
    % annotations
    xticks(ax,1:4); xticklabels(ax,posLabels);
    ylabel(ax,'% of trials','FontName','Arial','FontSize',10);
    title(ax,condNames{c},'FontName','Arial','FontSize',12,'FontWeight','bold');
    legend(ax, dsNames, 'Location','northwest','Box','off','FontSize',8);
    set(ax,'FontName','Arial','FontSize',10,'LineWidth',1);
end
xlabel(T,'Target Position','FontName','Arial','FontSize',10);
end




