function params = defaultCfg()
    %%%%%%%%%%%%%%
    %% Epoching %%
    %%%%%%%%%%%%%%
    params.fsamp = 512;
    params.epochSamples = -0.5*params.fsamp+1:1.0*params.fsamp;
    params.epochTime = params.epochSamples./params.fsamp;
    params.epochOnset = find(params.epochTime == 0);
    
    %%%%%%%%%%%%%%%%%%%%%
    %% Spectral Filter %%
    %%%%%%%%%%%%%%%%%%%%%
    params.spectralFilter.freqs = [1 30]; 
    params.spectralFilter.order = 2;  
    
    params.EOG.spectralFilter.freqs = [1 10];  
    params.EOG.spectralFilter.order = 2;  
    %%%%%%%%%%%%%%%
    %% Balancing %%
    %%%%%%%%%%%%%%%
    params.balance_iscompute = true;

    %%%%%%%%%%%%%%%%%%%%
    %% ROI Selection %%%
    %%%%%%%%%%%%%%%%%%%%
    params.roi = 'None'; % {'None', 'P/PO'}
    params.fisher_iscompute = true;

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Baseline Correction %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    params.baseline_iscompute = true;
    params.baseline_window = [-0.2, 0];

    %%%%%%%%%%%%%%%%%%%%
    %% Spatial Filter %%
    %%%%%%%%%%%%%%%%%%%%
    params.spatialFilter.type = 'xDAWN';  % {'CCA','xDAWN','None'}
    params.spatialFilter.time = round(0.1*params.fsamp)+1:round(0.5*params.fsamp);
    params.spatialFilter.time = params.spatialFilter.time + params.epochOnset;
    params.spatialFilter.nComp = 2;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Power Spectral Density %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params.psd.is_compute = true;
    params.psd.diff_iscompute = true;
    params.psd.type = 'stockwell';  % {'stockwell'}
    params.psd.roi = 'all'; % {'lIFG','rIFG','midfrontal','all'}
    params.psd.time = round(0.15*params.fsamp)+1:round(0.5*params.fsamp);
    params.psd.time = params.psd.time + params.epochOnset;
    params.psd.window = hanning(length(params.psd.time));
    params.psd.nfft  = 4*params.fsamp;
    params.psd.overlap = [];
    params.psd.freq_range = [8:1:14]; %[13:1:30] for beta

    %%%%%%%%%%%%%%
    %% Features %%
    %%%%%%%%%%%%%%
    params.features.erp_iscompute = true;
    params.features.diffwave_iscompute = true;
    params.statsfeatures.is_compute = false;

    %%%%%%%%%%%%%%%%%%%%%%
    %% Resampling Ratio %%
    %%%%%%%%%%%%%%%%%%%%%%
    params.resample.is_compute = true;
    params.resample.ratio = round(params.fsamp / 64);
    % params.resample.time = round(0.15*params.fsamp)+1:round(0.5*params.fsamp);
    params.resample.time = round(0.1*params.fsamp)+1:round(0.5*params.fsamp);
    % params.resample.time = round(0.1*params.fsamp)+1:round(0.3*params.fsamp);
    params.resample.time = params.resample.time + params.epochOnset;

    %%%%%%%%%%%%%%%%
    %% Classifier %%
    %%%%%%%%%%%%%%%%
    params.classify.is_normalize = true;
    params.classify.normtype = 'zscore'; % {'minmax','zscore'}
    params.classify.reduction.type = 'r2'; % {'pca', 'lasso', 'r2','None'} 
    params.classify.reduction.numfeats = 30;
    params.classify.reduction.pcaprct = 95;
    params.classify.type = 'linear'; % {'linear', 'diaglinear','SVM'}
      
end