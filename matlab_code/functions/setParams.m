% Configures various predefined parameters and saves it into a structure 
% variable 'cfg'. 
function params = setParams(header)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %% General Information %%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    params.fsamp = header.SampleRate;  

    params.eegChannels = 1:32; 
    params.eogChannels = 33:35;
    params.triggerChannel = 36;

    params.chanLabels = header.Label;


    params.plotOption = {'LineWidth', 2};
    params.plotColor = {
    [0.698, 0.133, 0.133],      % Red
    [0, 0, 1],                  % Blue
    [0.941, 0.502, 0.502],      % Light Red
    [0.6, 0.6, 1],              % Light Blue
    [0, 0.502, 0],              % Green
    [0, 0, 0],                  % Black
    [0.196, 0.804, 0.196],      % Light green
    [0.804, 0.361, 0.361],      % Red3 (Indian red)
    [0.196, 0.804, 0.196],      % Green3 (Lime green)
            };

    %%%%%%%%%%%%%%
    %% Epoching %%
    %%%%%%%%%%%%%%
    params.epochSamples = -0.5*params.fsamp+1:1.0*params.fsamp;
    params.epochTime = params.epochSamples./params.fsamp;
    params.epochOnset = find(params.epochTime == 0);
    
    %%%%%%%%%%%%%%%%%%%%%
    %% Spectral Filter %%
    %%%%%%%%%%%%%%%%%%%%%
    params.spectralFilter.freqs = [1 10]; 
    params.spectralFilter.order = 2;  
    
    params.EOG.spectralFilter.freqs = [1 10];  
    params.EOG.spectralFilter.order = 2;  

    %%%%%%%%%%%%%%%
    %% Balancing %%
    %%%%%%%%%%%%%%%
    params.balance_iscompute = false;

    %%%%%%%%%%%%%%%%%%%%
    %% ROI Selection %%%
    %%%%%%%%%%%%%%%%%%%%
    params.roi = 'ERP'; % {'None', 'P/PO'}
    params.fisher_iscompute = false;

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Baseline Correction %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    params.baseline_iscompute = true;
    params.baseline_window = [-0.2, 0];

    %%%%%%%%%%%%%%%%%%%%
    %% Spatial Filter %%
    %%%%%%%%%%%%%%%%%%%%
    params.spatialFilter.type = 'CCA';  % {'CCA','xDAWN','None'}
    params.spatialFilter.time = round(0.2*params.fsamp)+1:round(0.8*params.fsamp);
    params.spatialFilter.time = params.spatialFilter.time + params.epochOnset;
    params.spatialFilter.nComp = 3;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Power Spectral Density %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params.psd.is_compute = false;
    params.psd.diff_iscompute = false;
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
    params.features.diffwave_iscompute = false;
    params.statsfeatures.is_compute = false;

    %%%%%%%%%%%%%%%%%%%%%%
    %% Resampling Ratio %%
    %%%%%%%%%%%%%%%%%%%%%%
    params.resample.is_compute = true;
    params.resample.ratio = round(params.fsamp / 64);
    params.resample.time = round(0.2*params.fsamp)+1:round(0.8*params.fsamp);
    params.resample.time = params.resample.time + params.epochOnset;

    %%%%%%%%%%%%%%%%
    %% Classifier %%
    %%%%%%%%%%%%%%%%
    params.classify.is_normalize = true;
    params.classify.normtype = 'zscore'; % {'minmax','zscore'}
    params.classify.reduction.type = 'r2'; % {'pca', 'lasso', 'r2','None'} 
    params.classify.reduction.numfeats = 40;
    params.classify.reduction.pcaprct = 95;
    params.classify.type = 'linear'; % {'linear', 'diaglinear','SVM'}
      
end