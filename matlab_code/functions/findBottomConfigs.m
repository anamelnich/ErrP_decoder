function bottomConfigs = findBottomConfigs(T, measureName, decoderType, bottomN, subjectsToSearch)
% findBottomConfigs  Find worst parameter combos by lowest mean & low variability
%   bottomConfigs = findBottomConfigs(T, measureName, decoderType)
%   bottomConfigs = findBottomConfigs(T, measureName, decoderType, bottomN)
%   bottomConfigs = findBottomConfigs(T, measureName, decoderType, bottomN, subjectsToSearch)
%
% Inputs:
%  - T:               table with fields T.subject (string) and T.perf (struct)
%  - measureName:     e.g. 'auprc','kappa','accuracy','tpr','tnr'
%  - decoderType:     'bilateral', 'right', or 'left'
%  - bottomN:         number of bottom configs to return (default = 10)
%  - subjectsToSearch: cell array of subject IDs to include (default = all subjects in T)
%
% Output:
%  - bottomConfigs:   table with variables:
%       ConfigIndex  (1..K)
%       MeanVal      (mean of T.perf.(decoderType).(measureName))
%       StdVal       (std across selected subjects)
%       Config       (the cfg struct for that combo, drawn from first subject)

if nargin < 4 || isempty(bottomN)
    bottomN = 10;
end
if nargin < 5 || isempty(subjectsToSearch)
    subjectsToSearch = unique(T.subject, 'stable');
end
subjectsToSearch = subjectsToSearch(:);

% Determine number of subjects and configs
S = numel(subjectsToSearch);
rowsTot = height(T);
rows1 = find(strcmp(T.subject, subjectsToSearch{1}));
nConfigs = numel(rows1);
% Validate consistency
for s = 2:S
    if numel(find(strcmp(T.subject, subjectsToSearch{s}))) ~= nConfigs
        error('Subject %s has a different number of configurations.', subjectsToSearch{s});
    end
end

% Build data matrix [S x nConfigs]
data = nan(S, nConfigs);
for s = 1:S
    rows_s = find(strcmp(T.subject, subjectsToSearch{s}));
    for k = 1:nConfigs
        perf = T.perf(rows_s(k));
        if isfield(perf, decoderType) && isfield(perf.(decoderType), measureName)
            data(s,k) = perf.(decoderType).(measureName);
        end
    end
end

% Compute statistics
mu    = nanmean(data, 1);
sigma = nanstd(data, 0, 1);

% Assemble into table
tbl = table((1:nConfigs)', mu', sigma', T.cfg(1:nConfigs), ...
    'VariableNames', {'ConfigIndex','MeanVal','StdVal','Config'});

% Sort by mean ascending (worst) then std ascending (lowest variability)
tblSorted = sortrows(tbl, {'MeanVal','StdVal'}, {'ascend','ascend'});

% Return bottom N
nTake = min(bottomN, height(tblSorted));
bottomConfigs = tblSorted(1:nTake,:);

% Display results
fprintf('Bottom %d configurations for %s (%s):\n', nTake, measureName, decoderType);
for i = 1:nTake
    ci = bottomConfigs.ConfigIndex(i);
    fprintf('  #%d: mean=%.3f, std=%.3f, cfg =\n', ci, bottomConfigs.MeanVal(i), bottomConfigs.StdVal(i));
    disp(bottomConfigs.Config(i));
end
end
