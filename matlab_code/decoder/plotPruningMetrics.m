function plotPruningMetrics(iterations, accuracy, auprc, tpr, tnr, trialsRemaining, subjectID)
% PLOTPRUNINGMETRICS  Publication-quality plot of pruning metrics
%
%   plotPruningMetrics(iterations, accuracy, auprc, tpr, tnr, trialsRemaining, subjectID)
%
% Inputs:
%   iterations      – vector of iteration numbers (1×nIter)
%   accuracy        – accuracy at each iteration (1×nIter)
%   auprc           – AUPRC at each iteration (1×nIter)
%   tpr             – true positive rate at each iteration (1×nIter)
%   tnr             – true negative rate at each iteration (1×nIter)
%   trialsRemaining – number of trials left at each iteration (1×nIter)
%   subjectID       – string or char array, e.g. 'Subject 01'
%
% Produces a 3×2 grid of subplots (last slot shows best-AUPRC iteration), each with:
%   • thick lines (LineWidth=2)
%   • large markers (MarkerSize=8)
%   • FontSize=14
%   • grid on, box on
%   • dashed reference lines at y=0.5 and y=0.7 for the first four metrics
% And a super-title showing the subjectID.

  % Figure setup
  fig = figure('Units','inches', ...
               'Position',[1 1 8 10], ...
               'Color','w');
  metrics = { ...
    'Accuracy',           accuracy; 
    'AUPRC',              auprc; 
    'True Positive Rate', tpr; 
    'True Negative Rate', tnr; 
    'Trials Remaining',   trialsRemaining 
  };

  for k = 1:5
    subplot(3,2,k);
    y = metrics{k,2};
    plot(iterations, y, '-o', ...
         'LineWidth',2, ...
         'MarkerSize',8);
    title(metrics{k,1}, 'FontSize',16, 'FontWeight','bold');
    xlabel('Iteration','FontSize',14);
    ylabel(metrics{k,1},'FontSize',14);
    grid on; box on;
    set(gca, 'FontSize',12, 'LineWidth',1.2, 'TickDir','out');
    % add dashed reference lines for first four metrics
    if k <= 4
      hold on;
      yline(0.5,'--','LineWidth',1);
      yline(0.7,'--','LineWidth',1);
      hold off;
    end
  end

  % 6th subplot: metrics at best-AUPRC iteration
  [~, idxBest] = max(auprc);
  bestIter = iterations(idxBest);
  vals = [accuracy(idxBest), auprc(idxBest), tpr(idxBest), tnr(idxBest)];
  subplot(3,2,6);
  bar(vals, 'LineWidth',1.5);
  set(gca, 'XTick', 1:4, ...
           'XTickLabel', {'Acc','AUPRC','TPR','TNR'}, ...
           'FontSize',12, 'LineWidth',1.2);
  ylabel('Value','FontSize',14);
  yline(0.5,'--','LineWidth',1);
  yline(0.7,'--','LineWidth',1);
  ylim([0.4 0.8]);
  title(sprintf('Metrics at Iteration %d (Max AUPRC)', bestIter), 'FontSize',16, 'FontWeight','bold');
  grid on; box on;

  % overall super-title
  sg = sgtitle(sprintf('%s — Pruning Dynamics', subjectID));
  sg.FontSize = 18;
  sg.FontWeight = 'bold';
end



