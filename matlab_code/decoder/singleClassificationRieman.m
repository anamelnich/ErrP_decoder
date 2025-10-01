function [Ytest, posteriors] = singleClassificationRieman(decoder, eeg)

%% ---------------- Baseline Correction ---------------- %%
if decoder.baseline_iscompute
    baseline = mean(eeg(decoder.baseline_idx, :, :), 1);
    eeg = eeg - baseline; % 768 x 58 x 180
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
eeg = erpEpochs; % 768 x 7 x 180
%% --------- xDAWN ---------- %%
[nSamples, ~, nTrials] = size(eeg);
filtered = nan(nSamples, size(decoder.spatialFilter.erp, 2), nTrials);
    
for i = 1:nTrials
    filtered(:, :, i) = eeg(:, :, i) * decoder.spatialFilter.erp;
end
eeg = filtered;
eeg = eeg(decoder.resample.time(1:decoder.resample.ratio:end), :, :); % 26 x 2 x 180

%% --------- Riemannian Geometry ---------- %%

template_nd = decoder.riemann.template_nd; % 26 x 2
template_d   = decoder.riemann.template_d; % 26 x 2

% Augment data with templates
nComp    = size(eeg,2);
totalCh = 3*nComp;
augmentedEpochs_test = zeros(size(eeg,1), totalCh, size(eeg,3));
for i = 1:size(eeg,3)
    td = eeg(:,:,i);  % [nSamples×nCCA]  
    augmentedEpochs_test(:,:,i) = [td, template_nd,template_d]; % 26 x 6 x 180       
end

cov_matrices_test = estimateRiemannianCovaraince(augmentedEpochs_test); % 6 x 6 x 180
cov_matrices_test = real(cov_matrices_test);
epsilon = 1e-6;
for i = 1:size(cov_matrices_test,3)
    cov_matrices_test(:,:,i) = cov_matrices_test(:,:,i) + epsilon*eye(size(cov_matrices_test,1));
end
ref    = decoder.riemann.reference_train + epsilon*eye(size(decoder.riemann.reference_train)); % 6 x 6
cov_rb = Affine_transformation(cov_matrices_test, ref); % 6 x 6 x 180
[Ytest, posteriors] = mdm_test(cov_rb,{decoder.riemann.prototype_nd, decoder.riemann.prototype_d});
