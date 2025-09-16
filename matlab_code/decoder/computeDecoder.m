function [decoder, classifierEpochs] = computeDecoderRight(trainEpochs, trainLabels, params)
% computeDecoderLeft trains a classifier to detect left-distractor trials 
% using lateralized EEG features from left vs. right posterior electrodes.
%
% Output:
%   decoder           - trained model structure with metadata
%   classifierEpochs  - feature matrix used to train the model

%% ==================== Trial Selection ==================== %%
% if params.balance_iscompute
%     keepMask = trainLabels ~= 1;  
%     trainEpochs = trainEpochs(:, :, keepMask);
%     trainLabels = trainLabels(keepMask);
% end 

%% ==================== Baseline Correction ==================== %%
if params.baseline_iscompute
    baseline_window = params.baseline_window;
    baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
    baseline = mean(trainEpochs(baseline_idx, :, :), 1);
    trainEpochs = trainEpochs - baseline;
end 

%% ==================== ROI Selection ==================== %%

% Distractor class: distractor left (label 2)
% No distractor class: distractor right, no distractor (labels 1 or 0)
% trainLabels(trainLabels == 1) = 0;
% trainLabels(trainLabels == 2) = 1;

if isequal(params.roi, 'ERP')

    Electrodes  = {'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'FC1', 'FC2','C1','C2'};
    elIdx  = find(ismember(params.chanLabels, Electrodes));
    
    erpEpochs   = trainEpochs(:, elIdx, :);


elseif isequal(params.roi, 'None')
    erpEpochs   = trainEpochs(:, :, :);
    [leftIdx, rightIdx] = findLeftRight(params.chanLabels);
    
    if params.fisher_iscompute

        ERP_idx = fisher(erpEpochs,trainLabels,params);
        erpEpochs = erpEpochs(:,ERP_idx,:);
        fprintf('Top 10 ERP channels (Fisher score):\n');
        for i = 1:10
            fprintf('  %2d: %s\n', i, params.chanLabels{ERP_idx(i)});
        end

    end
end 


%% ==================== Power Spectral Density ==================== %%
if (params.psd.is_compute)
    if isequal(params.psd.roi, 'lIFG')
        electrodes = {'AF7','AF3','F5','F3','FC5','FC3','FT7','F7'};
    elseif isequal(params.psd.roi, 'rIFG')
        electrodes = {'AF8','AF4','F6','F4','FC6','FC4','FT8', 'F8'};
    elseif isequal(params.psd.roi, 'midfrontal')
        electrodes = {'AFZ','FZ','FCZ','F1','F2','FC2','FC1'};
    end
    if isequal(params.psd.roi,'all')
        elecIdx = find(ismember(params.chanLabels, params.chanLabels));
        psdEpochs = trainEpochs;
    else
        elecIdx = find(ismember(params.chanLabels, electrodes));
        psdEpochs = trainEpochs(:, elecIdx, :);
    end
    [psds, params] = compute_stockwell(psdEpochs, params);
    psds = squeeze(mean(abs(psds).^2, 1)); % 768 x 58 x 480
    if params.fisher_iscompute
        if params.psd.diff_iscompute
            psds_diff = psds(:,rightIdx,:) - psds(:,leftIdx,:);
            alphadiff_idx = fisher(psds_diff,trainLabels,params);
            psds_diff = psds_diff(:,alphadiff_idx,:); %768 x 10 x 480
            fprintf('Top 10 Alpha Difference Power channels (Fisher score):\n');
            for i = 1:10
                fprintf('  %2d: %s\n', i, newlabels{alphadiff_idx(i)});
            end
        end
        alpha_idx = fisher(psds,trainLabels,params);
        psds = psds(:,alpha_idx,:); %768 x 10 x 480
        fprintf('Top 10 Alpha Power channels (Fisher score):\n');
        for i = 1:10
            fprintf('  %2d: %s\n', i, params.chanLabels{alpha_idx(i)});
        end
        
    end
    psds = psds(params.resample.time(1:params.resample.ratio:end),:,:); 
    psds = reshape(psds, [size(psds,1)*size(psds,2) size(psds,3)]); % 260 x 480
    if params.psd.diff_iscompute
        psds_diff = psds_diff(params.resample.time(1:params.resample.ratio:end),:,:); 
        psds_diff = reshape(psds_diff, [size(psds_diff,1)*size(psds_diff,2) size(psds_diff,3)]); % 260 x 480
    end
    if any(isnan(psds))
        logicalIdx  = not(isnan(psds(:,1)));
        psds = psds(logicalIdx,:);
    end
end

%% ==================== Feature Extraction ==================== %%
% ERP features
if params.features.erp_iscompute
    [ERP_feats, ERPfilter, ERP_pca] = processFeatures(erpEpochs, trainLabels, params);
else
    ERP_feats = [];
    ERPfilter = 'na';
    ERP_pca = 'na';
end

% Difference wave features
if params.features.diffwave_iscompute
    [Diff_feats, DiffFilter,Diff_pca] = processFeatures(diffEpochs, trainLabels, params);
else
    Diff_feats = [];
    DiffFilter = 'na';
    Diff_pca = 'na';
end

% TFR wave features
if params.psd.is_compute
    if params.psd.diff_iscompute
        tfr_feats = cat(1,psds,psds_diff); % 520 x 480
    else
        tfr_feats = psds;
    end
else
    tfr_feats = [];
end


classifierEpochs = cat(1, ERP_feats, Diff_feats, tfr_feats);

if isempty(classifierEpochs)
    error('No features selected. Set at least one of the params.features flags to true.');
end

%% ==================== Dimensionality Reduction ==================== %%

% Normalization
if isequal(params.classify.normtype, 'minmax')
    maxVal = max(classifierEpochs, [], 2); % max value in each feature
    minVal = min(classifierEpochs, [], 2);
    normalize = @(x) (x - minVal) ./ (maxVal - minVal);
    classifierEpochs = normalize(classifierEpochs);
elseif isequal(params.classify.normtype, 'zscore')
    avg = mean(classifierEpochs,2);
    stdev = std(classifierEpochs,0,2);
    normalize = @(x)(x-avg)./stdev;
    classifierEpochs = normalize(classifierEpochs);
end 

% LASSO
if isequal(params.classify.reduction.type, 'lasso')
    lambdaMax = 0.1;
    Lambda = logspace(log10(0.001 * lambdaMax), log10(lambdaMax), 100);
    cvmodel = fitrlinear(classifierEpochs, trainLabels, 'ObservationsIn', 'columns', ...
        'Lambda', Lambda, 'KFold', 5, 'Learner', 'leastsquares', ...
        'Solver', 'sparsa', 'Regularization', 'lasso');
    mse = kfoldLoss(cvmodel);
    [~, idx] = min(mse);
    selectedLambda = Lambda(idx);

    modelLasso = fitrlinear(classifierEpochs, trainLabels, 'ObservationsIn', 'columns', ...
        'Lambda', selectedLambda, 'Learner', 'leastsquares', ...
        'Solver', 'sparsa', 'Regularization', 'lasso');
    
    keepIdx = modelLasso.Beta ~= 0;
    classifierEpochs = classifierEpochs(keepIdx, :);
    disp(['Number of features selected: ', num2str(sum(keepIdx))]);

elseif isequal(params.classify.reduction.type, 'r2')
    power = compute_r2(permute(classifierEpochs, [1 3 2]), trainLabels); 
    [~, keepIdx] = sort(power, 'descend');
    keepIdx = keepIdx(1:params.classify.reduction.numfeats);
    classifierEpochs = classifierEpochs(keepIdx, :);
end

%% ==================== Model Training ==================== %%

if strcmp(params.classify.type,'SVM')

    % 1) Define your mini‐grid of Cs
    C_values = [0.01, 0.1, 1, 10];
    bestLoss = Inf;
    bestC    = C_values(1);

    % 2) Cross‐validate to pick best C
    for c = C_values
        % create a 5‐fold CV SVM
        cvMdl = fitcsvm( ...
            classifierEpochs', trainLabels, ...
            'KernelFunction','linear', ...
            'Standardize',   false, ...
            'Prior',         'uniform', ...
            'BoxConstraint', c, ...
            'CrossVal',      'on', ...
            'KFold',         5 ...
        );
        loss = kfoldLoss(cvMdl);
        if loss < bestLoss
            bestLoss = loss;
            bestC    = c;
        end
    end
    fprintf('Chosen C from grid: %.4g (CV loss=%.4f)\n', bestC, bestLoss);

    % 3) Retrain on the full data with best C
    finalMdl = fitcsvm( ...
        classifierEpochs', trainLabels, ...
        'KernelFunction','linear', ...
        'Standardize',   false, ...
        'Prior',         'uniform', ...
        'BoxConstraint', bestC ...
    );

    % 4) Calibrate to get posterior probabilities
    %    Pass in X and Y since finalMdl is not cross‐validated
    modelRaw = fitSVMPosterior(finalMdl, classifierEpochs', trainLabels);

    %--- Wrap into a prediction function handle that returns [label, score] ---%
    % Note: predict(svmModel, Xnew) gives [predictedLabel, score], where
    % score(:,2) are the posterior probabilities for the positive class.
    model = @(Xnew) predict(modelRaw, Xnew);
    
elseif any(strcmp(params.classify.type, {'linear','diaglinear'}))
    modelRaw = fitcdiscr(classifierEpochs', trainLabels, ...
    'Prior', 'uniform', 'DiscrimType', params.classify.type);

    % Transform LDA output to probability using a sigmoid fit
    w = modelRaw.Coeffs(2,1).Linear;
    mu_coef = modelRaw.Coeffs(2,1).Const;
    distance = classifierEpochs' * w + mu_coef;
    p1 = 0.025; p2 = 1 - p1;
    b1 = -log((1 - p1) / p1) / prctile(distance, 100 * p1);
    b2 = -log((1 - p2) / p2) / prctile(distance, 100 * p2);
    b = (b1 + b2) / 2;
    model = @(x) 1 ./ (1 + exp(-b * (x * w + mu_coef)));
end


%% ==================== Store Decoder ==================== %%
decoder = struct();
decoder.Classes = modelRaw.ClassNames;
decoder.fsamp = params.fsamp;
decoder.epochOnset = params.epochOnset;
decoder.numFeatures = size(classifierEpochs, 1);
decoder.roi = params.roi;
decoder.fisher_iscompute = params.fisher_iscompute;
if params.fisher_iscompute
    decoder.fisher = struct();  
    if params.features.erp_iscompute
        decoder.fisher.erp = ERP_idx;
    end
    if params.features.diffwave_iscompute
        decoder.fisher.diff = diff_idx;
    end
    if params.psd.is_compute
        decoder.fisher.alpha = alpha_idx;
        decoder.fisher.alpha_diff = alphadiff_idx;
    end
end
decoder.classify = struct( ...
    'type', params.classify.type, ...
    'is_normalize', params.classify.is_normalize, ...
    'reduction', struct('type', params.classify.reduction.type) ...
);
if params.classify.is_normalize
    decoder.classify.funNormalize = normalize;
end
if ismember(params.classify.reduction.type, {'lasso', 'r2'})
    decoder.classify.keepIdx = keepIdx;
end
decoder.classify.applyPCA = struct( ...
        'erp', ERP_pca, ...
        'diff', Diff_pca ...
    );
decoder.classify.model = model;
decoder.resample = params.resample;
decoder.statsfeatures.is_compute = params.statsfeatures.is_compute;
decoder.features = struct( ...
    'erp_iscompute', params.features.erp_iscompute, ...
    'diffwave_iscompute', params.features.diffwave_iscompute ...
);

decoder.spatialFilter = struct( ...
    'erp', ERPfilter, ...
    'diff', DiffFilter ...
);
decoder.spatialFilter.type = params.spatialFilter.type;
decoder.elIdx = elIdx;
decoder.psd = params.psd;
decoder.psd.diff_iscompute = params.psd.diff_iscompute;
if params.psd.is_compute
    decoder.elecIdx = elecIdx;
end
decoder.baseline_iscompute = params.baseline_iscompute;
if params.baseline_iscompute
    decoder.baseline_idx = baseline_idx;
end 
decoder.balance_iscompute = params.balance_iscompute;
if exist('selectedLambda', 'var')
    decoder.lassoLambda = selectedLambda;
end 

end

%% ==================== Helper Functions ==================== %%
function [features, filterMatrix, applyPCA] = processFeatures(epochData, trainLabels, params)

filterMatrix = [];
applyPCA = [];

% Apply spatial filter
if strcmp(params.spatialFilter.type,'CCA')
    eeg = epochData(params.spatialFilter.time, :, :);
    filterMatrix = get_cca_spatialfilter(eeg, trainLabels);
    filterMatrix = filterMatrix(:, 1:params.spatialFilter.nComp);
    epochData = apply_spatialFilter(epochData, filterMatrix);
elseif strcmp(params.spatialFilter.type,'xDAWN')
    [filters, ~, ~] = xdawn(epochData, trainLabels, params.spatialFilter.nComp, params.spatialFilter.time);
    classes = unique(trainLabels);
    idx1    = find(classes==1);
    sr      = (idx1-1)*params.spatialFilter.nComp + (1:params.spatialFilter.nComp);
    filterMatrix = filters(sr, :)';    % [n_ch x n_comp]
    epochData = apply_spatialFilter(epochData, filterMatrix);
end

% Resample
if params.resample.is_compute
    resamps = epochData(params.resample.time(1:params.resample.ratio:end), :, :);
    features = reshape(resamps, [], size(epochData, 3));
elseif params.statsfeatures.is_compute
    epochData = epochData(params.resample.time, :, :); % 179 x 2 x 480
    avg = mean(epochData,1); % 1 x 2 comp x 480 trials
    variance = var(epochData,0,1); % 1 x 2 comp x 480 trials
    [peak_amp,peak_latency] = max(epochData,[],1); % 1 x 2 comp x 480 trials
    pos_area = sum(max(epochData, 0), 1); % 1 x 2 comp x 480 trials
    features = cat(1, avg, variance, peak_amp, peak_latency, pos_area); % 5 feat x 2 comp x 480 trial
    features = reshape(features, [], size(features, 3)); % 10 x 480
else
    features = [];
end

if isequal(params.classify.reduction.type, 'pca')
    [coeff, ~, ~, ~, explained, mu] = pca(features');
    numKeep = find(cumsum(explained) > params.classify.reduction.pcaprct, 1);
    fprintf('Number of PCA components: %d \n',numKeep)
    coeff = coeff(:, 1:numKeep);
    applyPCA = @(x) bsxfun(@minus, x', mu) * coeff;
    features = applyPCA(features)';
end


end

function output = apply_spatialFilter(data, filter)
% Applies spatial filter to each trial of data
[nSamples, ~, nTrials] = size(data);
nComp = size(filter, 2);
output = nan(nSamples, nComp, nTrials);
for i = 1:nTrials
    output(:, :, i) = data(:, :, i) * filter;
end
end
