% runAllSubjects.m
% subjects = {'e1','e2','e3','e4','e5','e6','e8','e10','e11','e12','e13','e14','e15'};
subjects = {'e1','e2','e3','e4','e5','e6','e8','e11','e12','e13','e14','e15'};
nSub     = numel(subjects);

% preallocate
corrs = nan(nSub,1);
aucs  = cell(nSub,1);
RTs   = cell(nSub,1);

cfg_input = [];  % if computeModel ignores cfg, otherwise pass your cfg struct

for i = 1:nSub
    subj = subjects{i};
    try
        fprintf('Processing %s…\n', subj);
        [corr_i, auc_i, RT_i] = computeModel(subj, cfg_input);
        corrs(i) = corr_i;

        % only keep up to the first 4 runs (or however many you have)
        nA = numel(auc_i);
        useA = min(nA,4);
        aucs{i} = auc_i(1:useA);

        nR = numel(RT_i);
        useR = min(nR,4);
        RTs{i} = RT_i(1:useR);

    catch ME
        warning('Failed on %s: %s', subj, ME.message);
        corrs(i) = NaN;
        aucs{i}  = [];
        RTs{i}   = [];
    end
end

% build a simple summary table for the correlations
T = table(subjects(:), corrs, ...
    'VariableNames', {'Subject','CorrAUCvsRT'});

% save everything
save('allSubjectResults8.mat', 'subjects','corrs','aucs','RTs');
writetable(T, 'allSubjectCorrs8.csv');

fprintf('Done! Results saved to allSubjectResults2.mat and allSubjectCorrs2.csv\n');

%% Plot correlation
% Prepare data
auc_all = vertcat(aucs{:});    % all run-means concatenated
RT_all  = vertcat(RTs{:});

auc_ctr = []; RT_ctr = [];
for i = 1:nSub
    a = aucs{i};  r = RTs{i};
    auc_ctr = [auc_ctr; a - mean(a)];
    RT_ctr  = [RT_ctr;  r - mean(r)];

end

% Compute correlations
[r_overall, p_overall] = corr(auc_all, RT_all, 'Type','Pearson');
[r_within,  p_within]  = corr(auc_ctr, RT_ctr,   'Type','Pearson');


%% 1) Overall AUC vs RT
figure('Units','inches','Position',[1 1 4 4],'PaperPositionMode','auto');
set(gcf,'Renderer','painters');
hold on;

% scatter
scatter(auc_all, RT_all, 60, 'o', ...
    'MarkerFaceColor',[0.2 0.6 0.8], ...
    'MarkerEdgeColor','k', ...
    'MarkerFaceAlpha',0.7, ...
    'LineWidth',0.5);

% fit line
p1 = polyfit(auc_all, RT_all,1);
x1 = linspace(min(auc_all), max(auc_all),100);
y1 = polyval(p1,x1);
plot(x1, y1, '-', 'LineWidth',1.5, 'Color','k');

% annotation
txt1 = sprintf('r = %.2f, p = %.3f', r_overall, p_overall);
text(mean(get(gca,'XLim')), max(get(gca,'YLim')) - 0.05*diff(get(gca,'YLim')), ...
     txt1, 'FontSize',12, 'HorizontalAlignment','center', ...
     'BackgroundColor','w','EdgeColor','k','Margin',4);

% axes styling
box on; grid off;
set(gca, 'FontName','Arial','FontSize',12,'LineWidth',1,'TickLength',[0.02 0.02]);
xlabel('Peak Amplitude (\muV)','FontSize',14,'FontName','Arial');
ylabel('Reaction Time (s)','FontSize',14,'FontName','Arial');
title('Overall Peak Amplitude vs RT','FontSize',14,'FontName','Arial');

%% 2) Within-subject (demeaned) AUC vs RT
figure('Units','inches','Position',[1 1 4 4],'PaperPositionMode','auto');
set(gcf,'Renderer','painters');
hold on;

scatter(auc_ctr, RT_ctr, 60, 'o', ...
    'MarkerFaceColor',[0.8 0.4 0.2], ...
    'MarkerEdgeColor','k', ...
    'MarkerFaceAlpha',0.7, ...
    'LineWidth',0.5);

p2 = polyfit(auc_ctr, RT_ctr,1);
x2 = linspace(min(auc_ctr), max(auc_ctr),100);
y2 = polyval(p2,x2);
plot(x2, y2, '-', 'LineWidth',1.5, 'Color','k');

txt2 = sprintf('r = %.2f, p = %.3f', r_within, p_within);
text(mean(get(gca,'XLim')), max(get(gca,'YLim')) - 0.05*diff(get(gca,'YLim')), ...
     txt2, 'FontSize',12, 'HorizontalAlignment','center', ...
     'BackgroundColor','w','EdgeColor','k','Margin',4);

box on; grid off;
set(gca, 'FontName','Times New Roman','FontSize',12,'LineWidth',1,'TickLength',[0.02 0.02]);
xlabel('\Delta Positive AUC (\muV\cdot ms)','FontSize',14,'FontName','Times New Roman');
ylabel('\Delta Reaction Time (ms)','FontSize',14,'FontName','Times New Roman');
title('Positive AUC vs Reaction Time Correlation','FontSize',14,'FontName','Times New Roman');

%% 3) Histogram of AUC (all trials)
figure('Units','inches','Position',[1 1 4 3],'PaperPositionMode','auto');
set(gcf,'Renderer','painters');

h = histogram(auc_all, 'BinWidth', 0.5, 'FaceColor',[0.2 0.6 0.8], 'EdgeColor','k');
xlabel('Peak Amplitude (\muV)','FontSize',14,'FontName','Arial');
ylabel('Count','FontSize',14,'FontName','Arial');

box on; grid off;
set(gca, 'FontName','Arial','FontSize',12,'LineWidth',1,'TickLength',[0.02 0.02]);
title('Distribution of Peak Amplitudes','FontSize',14,'FontName','Arial');
