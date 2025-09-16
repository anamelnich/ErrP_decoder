function [posterior, epoch] = singleClassification(decoder, eeg)
% singleClassificationRight applies a trained decoder to classify EEG data
% from a single trial using right vs. left posterior electrode features.
%
% Inputs:
%   decoder        - Struct containing trained model and feature parameters
%   eeg            - [time x channels x trials] EEG data
%
% Outputs:
%   posterior - Classifier output (probability or score)
%   epoch     - Feature vector used for classification

%% ---------------- Baseline Correction ---------------- %%
if decoder.baseline_iscompute
    baseline = mean(eeg(decoder.baseline_idx, :, :), 1);
    eeg = eeg - baseline;
end 
%% --------- ROI Extraction & Difference Wave ---------- %%

if isequal(decoder.roi, 'ERP')
    erpEpochs  = eeg(:, decoder.elIdx, :);

elseif isequal(decoder.roi, 'None')
    erpEpochs  = eeg(:, :, :);
   
    if decoder.fisher_iscompute
        if decoder.features.erp_iscompute
            erpEpochs = erpEpochs(:,decoder.fisher.erp,:);
        end
        if decoder.features.diffwave_iscompute
            diffEpochs = diffEpochs(:,decoder.fisher.diff,:);
        end
    end
end

%% --------- Power Spectral Density ---------- %%
if (decoder.psd.is_compute)
    psd_epoch = eeg(:, decoder.elecIdx,:);
    [psd, decoder] = compute_stockwell(psd_epoch,decoder);
    psd = squeeze(mean(abs(psd).^2, 1));
    if decoder.psd.diff_iscompute
        psd_diff = psd(:, decoder.rightElectrodeIndices, :) - psd(:, decoder.leftElectrodeIndices, :);
    end
    if decoder.fisher_iscompute
        psd = psd(:,decoder.fisher.alpha,:);
        if decoder.psd.diff_iscompute
            psd_diff = psd_diff(:,decoder.fisher.alpha_diff,:);
        end
    end
    psd = psd(decoder.resample.time(1:decoder.resample.ratio:end),:,:);
    [~, ~, n_trials] = size(psd);
    psd = reshape(psd, [size(psd,1)*size(psd,2) n_trials]);
    if decoder.psd.diff_iscompute
        psd_diff = psd_diff(decoder.resample.time(1:decoder.resample.ratio:end),:,:);
        [~, ~, n_trials] = size(psd_diff);
        psd_diff = reshape(psd_diff, [size(psd_diff,1)*size(psd_diff,2) n_trials]);
    end
end
    
%% ---------------- Feature Processing ---------------- %%
if decoder.features.erp_iscompute
    ERP_feats = processFeatures(erpEpochs, decoder, decoder.spatialFilter.erp, ...
        decoder.classify.applyPCA.erp);
else
    ERP_feats = [];
end

if decoder.features.diffwave_iscompute
    Diff_feats = processFeatures(diffEpochs, decoder, decoder.spatialFilter.diff,...
        decoder.classify.applyPCA.diff);
else
    Diff_feats = [];
end

% TFR wave features
if decoder.psd.is_compute
    if decoder.psd.diff_iscompute
        tfr_feats = cat(1,psd,psd_diff); % 520 x 480
    else
        tfr_feats = psd;
    end
else
    tfr_feats = [];
end

epoch = cat(1, ERP_feats, Diff_feats, tfr_feats);

%% ----------- Apply Dimensionality Reduction ----------- %%

if decoder.classify.is_normalize
    epoch = decoder.classify.funNormalize(epoch);
end

if ismember(decoder.classify.reduction.type, {'lasso', 'r2'})
    epoch = epoch(decoder.classify.keepIdx, :);
end

%% ------------------- Classification ------------------- %%
if strcmp(decoder.classify.type,'SVM')
    [~, score_all] = decoder.classify.model(epoch');
    posterior = score_all(:,2);  % P of class 1 (distractor class)
elseif any(strcmp(decoder.classify.type, {'linear','diaglinear'}))
    posterior = decoder.classify.model(epoch');
end

% if decoder.balance_iscompute
%     posterior(labels == 1) = NaN; 
% end 

end

%% ===================================================== %%
%% ================= Helper Functions ================== %%
%% ===================================================== %%

function features = processFeatures(eeg, decoder, filterMatrix,applyPCA)
% Applies spatial filtering and resampling to extract features

[nSamples, ~, nTrials] = size(eeg); % if no 3rd dimension, set to 1
if isequal(decoder.spatialFilter.type, 'None')
    filtered = eeg;
else
    filtered = nan(nSamples, size(filterMatrix, 2), nTrials); %768 x 2 x n trials
    
    for i = 1:nTrials
        filtered(:, :, i) = eeg(:, :, i) * filterMatrix; % 768 x 2 x n of trials
    end
end

% Resample
if decoder.resample.is_compute
    resamp = filtered(decoder.resample.time(1:decoder.resample.ratio:end), :, :);
    features = reshape(resamp, [], nTrials);
elseif decoder.statsfeatures.is_compute
    filtered = filtered(decoder.resample.time, :, :); % 179 x 2 x 480
    avg = mean(filtered,1); % 1 x 2 comp x 480 trials
    variance = var(filtered,0,1); % 1 x 2 comp x 480 trials
    [peak_amp,peak_latency] = max(filtered,[],1); % 1 x 2 comp x 480 trials
    pos_area = sum(max(filtered, 0), 1); % 1 x 2 comp x 480 trials
    features = cat(1, avg, variance, peak_amp, peak_latency, pos_area); % 5 feat x 2 comp x 480 trial
    features = reshape(features, [], size(features, 3)); % 10 x 480
else
    features = [];
end
if isequal(decoder.classify.reduction.type, 'pca')
    features = applyPCA(features)';
end

end
