function [performance, bestItrData] = iterativePrune(data, cfg, nIter)
%ITERATIVEPRUNE Perform iterative trial pruning and classifier evaluation
%   [performance, bestItrData] = iterativePrune(data, cfg, nIter)
%   data : struct with fields data, labels, file_id, eof, RT, tpos
%   cfg          : config struct with field balance_iscompute
%   nIter        : (optional) number of pruning iterations (default: 20)

if nargin < 3 || isempty(nIter)
    nIter = 20;
end

% Initialize variables
nFiles      = numel(data.eof);
masterMask  = true(numel(data.labels),1);
TPR   = nan(nIter,1);
TNR   = nan(nIter,1);
ACC   = nan(nIter,1);
AUPRC = nan(nIter,1);
thr   = nan(nIter,1);
Ntr   = nan(nIter,1);
maskMatrix = false(numel(data.labels), nIter);

for iter = 1:nIter
    fprintf('--- Iteration %d of %d ---\n', iter, nIter);

    % Prune data according to masterMask
    prunedData.data    = data.data(:, :, masterMask);
    prunedData.labels  = data.labels(masterMask);
    prunedData.file_id = data.file_id(masterMask);

    % Balance runs if requested
    if isfield(cfg,'balance_iscompute') && cfg.balance_iscompute
        mask = balanceRuns(prunedData);
        balancedData.data    = prunedData.data(:, :, mask);
        balancedData.labels  = prunedData.labels(mask);
        balancedData.file_id = prunedData.file_id(mask);
    else
        balancedData = prunedData;
    end

    % Cross-validation across files
    fprintf('Performing cross-validation...\n');
    post  = nan(numel(data.labels),1);
    for fileIdx = 1:nFiles
        trainIdx = balancedData.file_id ~= fileIdx;
        testIdx  = data.file_id == fileIdx;
        % Train decoder for right-distractor classification
        [decoderR, ~] = computeDecoder(...
            balancedData.data(:,:,trainIdx), ...
            balancedData.labels(trainIdx), cfg);
        % Classify held-out trials
        post(testIdx) = singleClassification(decoderR, ...
            data.data(:,:,testIdx));
    end
    data.posteriors = post;

    % Compute AUPRC
    [~, ~, ~, aucRight] = perfcurve(data.labels, ...
        data.posteriors, 1, 'Prior','uniform', 'xCrit','reca','yCrit','prec');

    % Find optimal threshold over a limited range
    range = linspace(0.35,0.65,61);
    [x,y,t,~,opt] = perfcurve(data.labels, ...
        data.posteriors, 1, 'Prior','uniform','TVals',range);
    threshold = t(x==opt(1) & y==opt(2));
    % threshold = 0.5;

    % Compute confusion metrics
    [tpr, tnr, acc] = printConfusionMatrix(data.labels, ...
        data.posteriors >= threshold);
    fprintf('Iteration %d: AUPRC = %.3f, TPR = %.3f, TNR = %.3f, ACC = %.3f, thr = %.3f\n', ...
        iter, aucRight, tpr, tnr, acc, threshold);

    % Store metrics
    TPR(iter)   = tpr;
    TNR(iter)   = tnr;
    ACC(iter)   = acc;
    AUPRC(iter) = aucRight;
    thr(iter)   = threshold;
    Ntr(iter)   = sum(masterMask);
    maskMatrix(:,iter) = masterMask;

    % Update masterMask by pruning low-confidence trials
    pruneMask = pruneTrialsMask(data.labels, data.posteriors, 0.5, 0.05, masterMask);
    masterMask = masterMask & pruneMask;
end

% Select best iteration based on AUPRC
[~, bestItr] = max(AUPRC);
bestMask = maskMatrix(:, bestItr);

% Assemble best-iteration data
bestItrData.data       = data.data(:,:,bestMask);
bestItrData.labels     = data.labels(bestMask);
bestItrData.file_id    = data.file_id(bestMask);
% bestItrData.RT         = data.RT(bestMask);
% bestItrData.tpos       = data.tpos(bestMask);
bestItrData.posteriors = data.posteriors(bestMask);

% Package performance metrics
performance.auprc      = AUPRC(bestItr);
performance.threshold  = thr(bestItr);
performance.accuracy   = ACC(bestItr);
performance.tpr        = TPR(bestItr);
performance.tnr        = TNR(bestItr);
performance.posteriors = bestItrData.posteriors;
performance.labels     = bestItrData.labels;
performance.file_id    = bestItrData.file_id;
performance.nTrials    = Ntr(bestItr);
performance.history    = struct('TPR',TPR,'TNR',TNR,'ACC',ACC,'AUPRC',AUPRC,'threshold',thr,'nTrials',Ntr);

end
% ================= Helper Function ================= %%
function [tpr,tnr,acc] = printConfusionMatrix(trueLabels, predictedLabels)
cm = confusionmat(logical(trueLabels), predictedLabels);
disp('Confusion Matrix (with labels):');
disp('--------------------------------');
disp('            Pred=0    Pred=1');
fprintf('True=0:       %3d       %3d\n', cm(1,1), cm(1,2));
fprintf('True=1:       %3d       %3d\n', cm(2,1), cm(2,2));
tnr = cm(1,1) / sum(cm(1,:));
tpr = cm(2,2) / sum(cm(2,:));
acc = sum(diag(cm)) / sum(cm(:));
fprintf('TNR: %.2f | TPR: %.2f | Accuracy: %.2f\n\n', tnr, tpr, acc);
end
