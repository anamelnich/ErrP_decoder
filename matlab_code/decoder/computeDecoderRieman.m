function decoder = computeDecoderRieman(trainEpochs, trainLabels, params)

%% ==================== Baseline Correction ==================== %%
 if params.baseline_iscompute
    baseline_window = params.baseline_window;
    baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
    baseline = mean(trainEpochs(baseline_idx, :, :), 1);
    trainEpochs = trainEpochs - baseline;
 end 
%% ==================== ROI Selection ==================== %%
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

if isequal(params.roi, 'P/PO')

    LeftElectrodes  = {'P1', 'P3', 'P5', 'P7', 'PO3', 'PO5', 'PO7'};
    RightElectrodes = {'P2', 'P4', 'P6', 'P8', 'PO4', 'PO6', 'PO8'};
    leftIdx  = find(ismember(params.chanLabels, LeftElectrodes));
    rightIdx = find(ismember(params.chanLabels, RightElectrodes));
    
    erpEpochs   = trainEpochs(:, rightIdx, :);
    diffEpochs  = trainEpochs(:, rightIdx, :) - trainEpochs(:, leftIdx, :);
 end
epochData = erpEpochs; % 768 x 7 x 120
%% ==================== xDAWN ==================== %%


[filters, ~, ~] = xdawn(epochData, trainLabels, params.spatialFilter.nComp, params.spatialFilter.time);
classes = unique(trainLabels);
idx1    = find(classes==1);
sr      = (idx1-1)*params.spatialFilter.nComp + (1:params.spatialFilter.nComp);
filterMatrix = filters(sr, :)';    % [n_ch x n_comp]
epochData = apply_spatialFilter(epochData, filterMatrix); % 768 x 2 x 120
%%
epochData = epochData(params.resample.time(1:params.resample.ratio:end), :, :); % 26 x 2 x 120

%% ==================== Riemannien Geometry ==================== %%
template_nd = mean(epochData(:,:,trainLabels == 0), 3); % 26 x 2
template_d = mean(epochData(:,:,trainLabels == 1), 3); % 26 x 2

decoder.riemann.template_nd = template_nd;
decoder.riemann.template_d = template_d;

%% Augment data with templates
augmentedEpochs = zeros(size(epochData, 1), size(epochData, 2) * 3, size(epochData, 3));  % 26 x 6 x 120
for i = 1:size(epochData, 3)
    % Extract the current trial
    trial_data = epochData(:,:,i); 
    % Concatenate both templates along the channel dimension (axis 2)
    augmented_data = [trial_data, template_nd, template_d];  
    % Store the augmented trial
    augmentedEpochs(:,:,i) = augmented_data;           
end

%% Compute covariance matrix
cov_matrices = estimateRiemannianCovaraince(augmentedEpochs); % 6 x 6 x 120
% Ensure the covariance matrices are real
cov_matrices = real(cov_matrices);
% Regularize to ensure positive definiteness
epsilon = 1e-6;  % Small regularization constant
for i = 1:size(cov_matrices, 3)
    cov_matrices(:, :, i) = cov_matrices(:, :, i) + epsilon * eye(size(cov_matrices, 1));
end

%% Compute reference matrix
reference_matrix = riemann_mean(cov_matrices); 
% Ensure the reference matrix is real
reference_matrix = real(reference_matrix);
% Regularize to ensure positive definiteness
reference_matrix = reference_matrix + epsilon * eye(size(reference_matrix, 1)); % 6 x 6
decoder.riemann.reference_train = reference_matrix;           

%% Rebias all trials using the reference matrix
cov_rebias_all = Affine_transformation(cov_matrices, reference_matrix);
% Extract covariance matrices for correct and error trials
cov_rebias_nd = cov_rebias_all(:,:,trainLabels == 0);  % Correct trials
cov_rebias_d = cov_rebias_all(:,:,trainLabels == 1);    % Error trials
% Compute prototypes 
prototype_nd = riemann_mean(cov_rebias_nd);  
prototype_d = riemann_mean(cov_rebias_d);  

% Ensure the prototypes are real
prototype_nd = real(prototype_nd);
prototype_error = real(prototype_d);
% Regularize to ensure positive definiteness
prototype_nd = prototype_nd + epsilon * eye(size(prototype_nd, 1)); % 6 x 6
prototype_d = prototype_d + epsilon * eye(size(prototype_d, 1)); % 6 x 6
% Store prototypes
decoder.riemann.prototype_nd = prototype_nd;
decoder.riemann.prototype_d = prototype_d;

decoder.baseline_idx = baseline_idx;

decoder.spatialFilter.erp = filterMatrix;
decoder.resample = params.resample;
decoder.baseline_iscompute = params.baseline_iscompute;
decoder.roi = params.roi;
decoder.elIdx = elIdx;
end
    
%% spatial filter helper function
function data_output = apply_spatialFilter(data_input, filter_matrix)
        [n_samples, ~, n_trials] = size(data_input);
        
        data_output = nan(n_samples, size(filter_matrix,2), n_trials);
        for i_trial = 1:n_trials
            data_output(:,:,i_trial) = data_input(:,:,i_trial) * filter_matrix;
        end
end

