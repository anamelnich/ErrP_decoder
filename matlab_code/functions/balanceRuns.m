function keepMask = balanceRuns(trainingData)
% BALANCERUNSMASK  Return a mask of trials to keep, balancing runs
%
%   keepMask = BALANCERUNSMASK(trainingData) returns a logical vector
%   of length numel(trainingData.labels).  Within each run, it
%   downsamples right‐ and left‐distractor trials to the same count
%   (the smaller of the two), and then samples that many no‐distractor
%   trials.  All other trials get masked out (false).

  % preallocate
  nTrialsTotal = numel(trainingData.labels);
  keepMask     = false(nTrialsTotal,1);

  uniqueRuns = unique(trainingData.file_id);
  for i = 1:numel(uniqueRuns)
    runID       = uniqueRuns(i);
    runMask     = (trainingData.file_id == runID);
    runIdxGlobal= find(runMask);             % indices into the full trial vector
    labelsRun   = trainingData.labels(runMask);

    % find global indices of each condition
    leftDist    = runIdxGlobal(labelsRun == 1);
    noDist      = runIdxGlobal(labelsRun == 0);

    % decide how many to keep per distractor side
    nKeep = min(numel(leftDist), numel(noDist));

    % random sample
    sampL       = randsample(leftDist,  nKeep);
    sampN       = randsample(noDist,    nKeep);

    % mark these as true
    keepMask([sampL; sampN]) = true;
  end
end
% 
% function trainingData = balanceRuns(trainingData)
% % BALANCERUNS  Balance distractor vs. no-distractor trials within each run
% %   trainingData = BALANCERUNS(trainingData, nTrials) downsamples no-distractor
% %   trials to at most nTrials per run, while keeping all distractor trials.
% %
% % Inputs:
% %   trainingData : struct with fields:
% %       .data    (features × samples × trials)
% %       .labels  (trials×1)
% %       .file_id (trials×1), run indices for each trial
% %   nTrials      : desired number of no-distractor trials to sample per run
% %
% % Output:
% %   trainingData : same struct, but with balanced .data, .labels, and .file_id
% 
% uniqueRuns = unique(trainingData.file_id);
% nFiles     = numel(uniqueRuns);
% 
% balancedData   = [];
% balancedLabels = [];
% balancedFileId = [];
% 
% for i = 1:nFiles
%     runID    = uniqueRuns(i);
%     runMask  = (trainingData.file_id == runID);
% 
%     dataRun   = trainingData.data(:, :, runMask);
%     labelsRun = trainingData.labels(runMask);
%     fileIdRun = trainingData.file_id(runMask);
% 
%     distrIdx   = find(labelsRun == 1);
%     distlIdx   = find(labelsRun == 2);
%     noDistIdx = find(labelsRun == 0);
% 
%     nTrials = min(length(distrIdx),length(distlIdx));
% 
%     sampDistR  = randsample(distrIdx, nTrials);
%     sampDistL  = randsample(distlIdx, nTrials);
%     sampNoDist  = randsample(noDistIdx, nTrials);
% 
%     keepIdx     = sort([sampDistR; sampDistL; sampNoDist]);
% 
%     balancedData   = cat(3, balancedData,   dataRun(:, :, keepIdx));
%     balancedLabels = [balancedLabels;       labelsRun(keepIdx)];
%     balancedFileId = [balancedFileId;       fileIdRun(keepIdx)];
% end
% 
% trainingData.data    = balancedData;
% trainingData.labels  = balancedLabels;
% trainingData.file_id = balancedFileId;
% end

