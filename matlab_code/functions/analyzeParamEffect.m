function summaryTable = analyzeParamEffect(T, parameterPath, measureName, decoderType, subjectsToInclude)
% analyzeParamEffect  Assess how a parameter impacts a performance metric
%   summaryTable = analyzeParamEffect(T, parameterPath, measureName, decoderType)
%   summaryTable = analyzeParamEffect(..., subjectsToInclude)
%
% Inputs:
%  - T:                table with T.subject, T.cfg, T.perf
%  - parameterPath:    e.g. 'classify.type' or 'psd.is_compute'
%  - measureName:      e.g. 'auprc','kappa','accuracy','tpr','tnr'
%  - decoderType:      'bilateral','right','left'
%  - subjectsToInclude (optional): cell of subject IDs
%
% Output:
%  - summaryTable: table with columns:
%       ParamValue  unique parameter values
%       Mean        mean metric across subjects
%       Std         std metric across subjects
%       Count       number of subjects with data

if nargin<5 || isempty(subjectsToInclude)
    subjectsToInclude = unique(T.subject,'stable');
end
subjectsToInclude = subjectsToInclude(:);

% Filter rows by subject
mask = ismember(T.subject, subjectsToInclude);
rows = find(mask);
n = numel(rows);
paramVals = cell(n,1);
measureVals = nan(n,1);

toks = strsplit(parameterPath, '.');
% Extract parameter and measure
for i=1:n
    perf = T.perf(rows(i));
    % get measure
    if isfield(perf,decoderType) && isfield(perf.(decoderType),measureName)
        measureVals(i) = perf.(decoderType).(measureName);
    else
        measureVals(i) = NaN;
    end
    % get cfg value
    cfg = T.cfg(rows(i));
    val = cfg;
    for t=1:numel(toks)
        if isstruct(val) && isfield(val,toks{t})
            val = val.(toks{t});
        else
            val = [];
            break;
        end
    end
    paramVals{i} = val;
end

% Determine unique parameter values
% Check type: all char or all numeric/logical
isChar = all(cellfun(@(x) ischar(x), paramVals));
isNum  = all(cellfun(@(x) (isnumeric(x)||islogical(x)) && ~isempty(x), paramVals));
if isChar
    [uniqVals,~,ic] = unique(paramVals,'stable');
elseif isNum
    dataNum = cell2mat(paramVals);
    [uniqVals,~,ic] = unique(dataNum);
else
    error('Parameter values must be uniformly char or numeric/logical');
end
m = numel(uniqVals);

% Compute stats for each unique value
means = nan(m,1);
stdev = nan(m,1);
counts = zeros(m,1);
for j=1:m
    idx = (ic==j);
    vals = measureVals(idx);
    means(j)  = nanmean(vals);
    stdev(j)  = nanstd(vals);
    counts(j) = sum(~isnan(vals));
end

% Assemble table
summaryTable = table(uniqVals,means,stdev,counts,'VariableNames',{'ParamValue','Mean','Std','Count'});

% Plot bar + errorbar
figure;
if isChar
    cats = categorical(uniqVals);
else
    cats = uniqVals;
end
bar(cats, means, 'FaceColor',[0.2 0.6 0.5]); hold on;
errorbar(cats, means, stdev, 'k.', 'LineWidth',1.5);
hold off;
xlabel(parameterPath);
ylabel(measureName);
title(sprintf('Effect of %s on %s (%s)',parameterPath,measureName,decoderType));
grid on;
end

