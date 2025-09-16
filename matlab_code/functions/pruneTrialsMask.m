function pruneMask = pruneTrialsMask(labels, post, thr, pctRemove, master)
  
  N = numel(labels);
  pruneMask = true(N,1);

  low_conf  = 0.2;
  high_conf = 0.8;
  classes   = unique(labels);

  for c = classes(:)'
    % only consider still-active trials of this class
    idx_c = find(master & labels==c);
    n_c   = numel(idx_c);
    if n_c==0, continue; end

    p_c = post(idx_c);

    % numbers to drop (5% of current pool)
    nMis = ceil(pctRemove * n_c);
    nThr = ceil(pctRemove * n_c);

    % 1) high-confidence misclassifications
    if c==1
      misIdx = idx_c(p_c < low_conf);
    else
      misIdx = idx_c(p_c > high_conf);
    end
    if ~isempty(misIdx)
      confMis = abs(post(misIdx) - thr);
      [~, ord] = sort(confMis, 'descend');
      drop1 = misIdx(ord(1:min(nMis, numel(ord))));
    else
      drop1 = [];
    end

    % 2) near-threshold trials
    distThr = abs(p_c - thr);
    [~, ord2] = sort(distThr, 'ascend');
    drop2 = idx_c(ord2(1:min(nThr, numel(ord2))));

    % mark for pruning
    pruneMask(unique([drop1; drop2])) = false;
  end
end
