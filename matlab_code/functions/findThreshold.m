function threshold = findThreshold(TPR,FPR,t,minTPR)

if nargin < 4 || isempty(minTPR), minTPR = 0.55; end 

TNR = 1 - FPR;

% Feasible set: meet sensitivity floor
feasible = TPR >= minTPR;

if ~any(feasible)
    % If no point meets the floor, pick the one with highest TPR, then highest TNR
    [~, iBestTPR] = max(TPR);
    ties = find(TPR == TPR(iBestTPR));
    [~, k] = max(TNR(ties));
    bestIdx = ties(k);
else
    % Among feasible points, pick the one with max TNR (highest specificity)
    [~, bestIdx] = max(TNR(feasible));
    idxs = find(feasible);
    bestIdx = idxs(bestIdx);
end

threshold = t(bestIdx);
end
% TNR = 1 - FPR;
% 
% diffs = abs(TPR - TNR);
% BA    = 0.5 * (TPR + TNR);        % balanced accuracy
% 
% minDiff = min(diffs);
% cand    = find(diffs <= minDiff + 1e-12);
% [~, k]  = max(BA(cand));
% bestIdx = cand(k);
% 
% threshold = t(bestIdx);
% end