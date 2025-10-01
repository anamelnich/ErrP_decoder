function threshold = findThreshold(TPR,FPR,t)

TNR = 1 - FPR;

diffs = abs(TPR - TNR);
BA    = 0.5 * (TPR + TNR);        % balanced accuracy

minDiff = min(diffs);
cand    = find(diffs <= minDiff + 1e-12);
[~, k]  = max(BA(cand));
bestIdx = cand(k);

threshold = t(bestIdx);
end