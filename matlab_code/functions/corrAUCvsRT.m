function [rho, pval, runValue, runRT] = corrAUCvsRT(data, params)
% [rho, pval, runValue, runRT] = corrAUCvsRT_byRun(data, params)
%   Computes per‐trial AUC/peak values and RTs, then averages both within
%   each run and returns the Pearson correlation across runs.
%
% Inputs:
%   data.data      - EEG matrix (time × channels × trials)
%   data.labels    - trial labels (1 or 2; 0 = ignore)
%   data.RT        - reaction times (1×trials)
%   data.file_id   - run index per trial (1×trials)
%   params         - struct with fields:
%     .baseline_window
%     .epochTime
%     .resample.time
%     .chanLabels
%
% Outputs:
%   rho            - Pearson’s r between runValue and runRT
%   pval           - two‐tailed p‐value
%   runValue       - vector of mean ‘value’ per run (nRuns×1)
%   runRT          - vector of mean RT per run (nRuns×1)

eeg    = data.data;
labels = data.labels;
RT     = data.RT;
runIdx = data.file_id;

%% 1) Baseline correction
bw   = params.baseline_window;
tidx = find(params.epochTime >= bw(1) & params.epochTime <= bw(2));
baseline = mean(eeg(tidx,:,:), 1);
eeg = eeg - baseline;

%% 2) Define ROI channels
Left  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
Right = {'P2','P4','P6','P8','PO4','PO6','PO8'};
Lidx  = find(ismember(params.chanLabels, Left));
Ridx  = find(ismember(params.chanLabels, Right));

%% 3) Compute single-trial values
nTrials = size(eeg,3);
value   = nan(nTrials,1);
valid   = labels~=0;

dt = 1000/params.fsamp; % time step in ms for positive area calc

for i = find(valid)'
    waveL = squeeze(mean(eeg(:,Lidx,i),2)); % 768 x 1
    waveR = squeeze(mean(eeg(:,Ridx,i),2));
    if labels(i)==1
        diffW = waveL - waveR;
    else
        diffW = waveR - waveL; %768 x 1 
    end
    tw = diffW(params.resample.time); % 205 x 1
    % value(i) = max(tw);    
    % value(i) = sum(max(tw,0));
    pd_pos = max(tw,0);
    value(i) = sum(pd_pos) * dt; 
end

% keep only valid trials
value = value(valid); % 240 x 1
RTv   = RT(valid);
runs  = runIdx(valid);

%% 4a) Remove RT outliers (as before)
rtMean = mean(RTv);
rtStd  = std(RTv);
rtBad  = RTv < 100 | RTv > (rtMean + 3*rtStd);
value(rtBad) = [];
RTv(rtBad)   = [];
runs(rtBad)  = [];

% 4b) Now remove peak‐amplitude outliers
valMean = mean(value);
valStd  = std(value);
valBad  = value > (valMean + 3*valStd);
value(valBad) = [];
RTv(valBad)   = [];
runs(valBad)  = [];

%% 5) Compute per‐run averages
uniqueRuns = unique(runs);
nRuns      = numel(uniqueRuns);
runValue   = nan(nRuns,1);
runRT      = nan(nRuns,1);

for k = 1:nRuns
    thisRun = uniqueRuns(k);
    idxRun  = runs==thisRun;
    runValue(k) = mean(value(idxRun));
    runRT(k)    = mean(RTv(idxRun));
end

%% 6) Correlate run‐means
[rho, pmat] = corr(runValue, runRT, 'Type','Pearson');
pval = pmat;

%% 7) Plot
%— Create figure at a fixed size (e.g. 4×4 inches) and high DPI —%
figure('Units','inches','Position',[1 1 4 4],'PaperPositionMode','auto');
set(gcf,'Renderer','painters');  % vector graphics

%— Scatter with semi-transparent, colored markers —%
hSc = scatter(runValue, runRT, 75, 'o', ...
    'MarkerFaceColor',[0.2 0.4 0.8], ...
    'MarkerEdgeColor','k', ...
    'MarkerFaceAlpha',0.7, ...
    'LineWidth',0.5);
hold on;

%— Fit and plot a least-squares line —%
p = polyfit(runValue, runRT,1);
xFit = linspace(min(runValue),max(runValue),100);
yFit = polyval(p,xFit);
hLn = plot(xFit, yFit, '-', ...
    'Color','k', ...
    'LineWidth',1.5);

%— Annotation of r and p on the plot —%
txt = sprintf('r = %.2f, p = %.3f', rho, pval);
xPos = mean(get(gca,'XLim'));
yPos = max(get(gca,'YLim')) - 0.05*diff(get(gca,'YLim'));
text(xPos, yPos, txt, ...
    'FontSize',12, ...
    'HorizontalAlignment','center', ...
    'BackgroundColor','w', ...
    'EdgeColor','k', ...
    'Margin',4);

%— Polish axes —%
box on;
grid off;
set(gca, ...
    'FontName','Arial', ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'TickLength',[0.02 0.02]);

xlabel('Peak Amplitude (\muV)', ...
    'FontName','Arial', ...
    'FontSize',14, ...
    'Interpreter','tex');
ylabel('Reaction Time (s)', ...
    'FontName','Arial', ...
    'FontSize',14, ...
    'Interpreter','tex');

end
