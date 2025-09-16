function [meanD, meanND, rtD_mean, rtND_mean] = computeDiffWave(eeg, labels, RT, params)
% COMPUTEDIFFWAVE  Compute mean diff-waves + mean RTs
%
% [meanD, meanND, rtD_mean, rtND_mean] = computeDiffWaveAndRT(eeg, labels, RT, params)
%
% Inputs:
%   eeg       - [time×chan×trial] epoched, raw (will be baseline-corrected here)
%   labels    - [1×trial]  0=no-distractor, 1=left-dist, 2=right-dist
%   RT        - [1×trial]  reaction times
%   params    - struct with fields:
%     .baseline_window  (e.g. [-200 0])
%     .epochTime        (time vector, length = size(eeg,1))
%     .chanLabels       (cell array, length = size(eeg,2))
%
% Outputs:
%   meanD      - [time×1]  mean distractor diff-wave
%   meanND     - [time×1]  mean no-distractor diff-wave (random sign)
%   rtD_mean   - scalar    mean RT on distractor trials
%   rtND_mean  - scalar    mean RT on no-distractor trials

%% 1) Baseline correction
bwIdx    = params.epochTime >= params.baseline_window(1) & ...
           params.epochTime <= params.baseline_window(2);
baseline = mean(eeg(bwIdx,:,:), 1);    % [1×chan×trial]
eeg       = eeg - baseline;            % subtract

%% 2) Define ROI channels
LeftCh  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightCh = {'P2','P4','P6','P8','PO4','PO6','PO8'};
Lidx    = find(ismember(params.chanLabels, LeftCh));
Ridx    = find(ismember(params.chanLabels, RightCh));

%% 3) Build diff-waves & collect RTs
nTrials    = size(eeg,3);
diffD      = [];    % will be [time×nDistTrials]
diffND     = [];    % [time×nNoDistTrials]
RT_D       = [];
RT_ND      = [];

for i = 1:nTrials
    wL = squeeze(mean(eeg(:,Lidx,i),2));
    wR = squeeze(mean(eeg(:,Ridx,i),2));
    switch labels(i)
      case {1,2}  % any distractor
        if labels(i)==1
          d = wL - wR;
        else
          d = wR - wL;
        end
        diffD(:,end+1) = d;
        % try to grab RT, but if it errors just leave RT_D empty
        try
          RT_D(end+1) = RT(i);
        catch ME
          warning('Skipping RT_D for trial %d: %s', i, ME.message);
          RT_D = [];
        end

      case 0  % no-distractor: random sign
        if rand < 0.5
          d = wL - wR;
        else
          d = wR - wL;
        end
        diffND(:,end+1) = d;
        try
          RT_ND(end+1) = RT(i);
        catch ME
          warning('Skipping RT_ND for trial %d: %s', i, ME.message);
          RT_ND = [];
        end
    end
end

%% 4) Compute means
meanD     = mean(diffD, 2);
meanND    = mean(diffND,2);
if ~isempty(RT_D)
  rtD_mean = mean(RT_D);
else
  rtD_mean = [];
end

if ~isempty(RT_ND)
  rtND_mean = mean(RT_ND);
else
  rtND_mean = [];
end
end

