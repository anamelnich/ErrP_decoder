function dataset = preprocessDataset(dataset, cfg,fname,trigType)
    if ~isfield(dataset, 'data') || isempty(dataset.data)
        warning('%s is empty or ''data'' field is missing. Skipping trigger extraction...',fname);
        return;  
    end
    % Extract trigger and EEG data
    dataset.trigger = dataset.data(:, cfg.triggerChannel);
    dataset.EOG = dataset.data(:, cfg.eogChannels);
    dataset.data = dataset.data(:, cfg.eegChannels);
    

    % Compute index
    if contains(fname, 'eogcalibration')
        dataset.index = computeIndexEOG(dataset.trigger);
    else
        dataset.index = computeIndex(dataset.trigger, trigType);
    end
end