function stat = clusterPermTest(allD, allND, t, timeWindow, clusteralpha, nPerm)
% CLUSTERPERMTEST  Two‐sided cluster‐based permutation test (time‐only)
%
% stat = clusterPermTest(allD, allND, t, timeWindow, clusteralpha, nPerm)
%
% (Inputs/Outputs unchanged from before.)

%% 1) Prep
[nSub, nT] = size(allD);
diffMat    = allD - allND;            
df         = nSub - 1;
idxWin     = t >= timeWindow(1) & t <= timeWindow(2);

%% 2) Observed t‐values
m_diff     = mean(diffMat, 1);
s_diff     = std(diffMat, 0, 1);
stat.t_obs = m_diff ./ (s_diff./sqrt(nSub));

%% 3) Threshold
stat.thresh = tinv(1 - clusteralpha/2, df);

%% 4) Observed clusters
obsMask = abs(stat.t_obs) > stat.thresh & idxWin;
stat.clusterIdx = findClusters(obsMask);
nClust = numel(stat.clusterIdx);
stat.clusterStat = zeros(1,nClust);
for k = 1:nClust
    idx = stat.clusterIdx{k};
    stat.clusterStat(k) = sum(abs(stat.t_obs(idx)));
end

%% 5) Permutation distribution
stat.tMaxPerm = zeros(1,nPerm);
for p = 1:nPerm
    signs = (rand(nSub,1)>0.5)*2 - 1;
    permMat = diffMat .* signs;
    m_p = mean(permMat,1);
    s_p = std(permMat,0,1);
    t_p = m_p ./ (s_p./sqrt(nSub));
    permMask = abs(t_p) > stat.thresh & idxWin;
    cl = findClusters(permMask);
    if isempty(cl)
        stat.tMaxPerm(p) = 0;
    else
        cs = cellfun(@(c) sum(abs(t_p(c))), cl);
        stat.tMaxPerm(p) = max(cs);
    end
end

%% 6) Cluster‐level p‐values (corrected)
stat.pCluster = nan(1, nClust);
for k = 1:nClust
    % proportion of permuted max‐stats >= observed clusterStat(k)
    stat.pCluster(k) = (sum(stat.tMaxPerm >= stat.clusterStat(k)) + 1) ...
                       / (nPerm + 1);
end

%% 7) Build a mask of significant time points
stat.mask = false(1,nT);
for k = find(stat.pCluster < clusteralpha)
    stat.mask(stat.clusterIdx{k}) = true;
end
end

%% Helper to find contiguous true runs
function clusters = findClusters(maskVec)
    d = diff([0, maskVec, 0]);
    starts = find(d==1);
    ends   = find(d==-1)-1;
    clusters = cell(1,numel(starts));
    for i = 1:numel(starts)
        clusters{i} = starts(i):ends(i);
    end
end

