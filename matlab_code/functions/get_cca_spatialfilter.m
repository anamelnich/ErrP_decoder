function spatialFilter = get_cca_spatialfilter(dataEpochs, dataLabels)
% dataEpochs - samples x channels x trials
% dataLabels - trials

concat_data = [];
concat_ga = [];
classes = unique(dataLabels);
for iClass = 1:length(classes)
    exClass = classes(iClass);
    exEpochs = dataEpochs(:, :, dataLabels == exClass);
    [n_samples,n_channels,n_trials] = size(exEpochs);        

    exEpochs = permute(exEpochs, [2 1 3]); %channel, sample, trial
    ga_data = mean(exEpochs, 3); %channel x sample 
    
    ex_epochs = reshape(exEpochs,[n_channels n_samples*n_trials]); % channel x sample*trial
    ga_data = repmat(ga_data,[1 n_trials]); % channel x sample*trial
    
    concat_data = cat(2, concat_data, ex_epochs);
    concat_ga = cat(2, concat_ga, ga_data);
end

[spatialFilter,~] =  canoncorr(concat_data', concat_ga'); %spatial filter will maximize correlation between data and grand average of each class

