function checkFeedback(subjectID)

%%%%%%%%%%%%%%%%%%%%
%% Initialization %%
%%%%%%%%%%%%%%%%%%%%
clearvars -except subjectID;
close all; clc; rng('default');

addpath(genpath('../functions'));
addpath(genpath('../decoder'));

%%%%%%%%%%%%%%%%%%
%% Load Dataset %%
%%%%%%%%%%%%%%%%%%
dataPath = [pwd '/../../data/'];
dataInfo = dir([dataPath '/' subjectID '_20*']);

disp(['Loading the data from ' dataInfo.name]);
[~, ~, feedback] = loadData([dataInfo.folder '/' dataInfo.name '/*_DecNef_*']);

load('decoder');

params = setParameters(feedback.header.SampleRate);
feedback.index = computeIndexFeedback(feedback.data(:, params.triggerChannel));

posterior_log = cell(0);
for i_trial = 1:length(feedback.index.start_index)
    ex_start = feedback.index.start_index(i_trial);
    if (i_trial == length(feedback.index.start_index))
        ex_end = size(feedback.data, 1);
    else
        ex_end = feedback.index.start_index(i_trial + 1);
    end
    ex_logical = ex_start <= feedback.header.EVENT.POS & feedback.header.EVENT.POS <= ex_end;
    posterior_log{i_trial} = [feedback.header.EVENT.POS(ex_logical), feedback.header.EVENT.TYP(ex_logical)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preprocess EEG signals %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
feedback.data(:, [decoder.eegChannels decoder.eogChannels]) = filter(decoder.spectralFilter.b, decoder.spectralFilter.a, feedback.data(:, [decoder.eegChannels, decoder.eogChannels]));
feedback.data(:, params.eegChannels) = feedback.data(:, params.eegChannels) - feedback.data(:, params.eogChannels) * decoder.eogFilter;

%%%%%%%%%%%%%%%%
%% Simulation %%
%%%%%%%%%%%%%%%%
simulation_log = cell(0);
window_length = round(0.8*decoder.fsamp);
for i_trial = 1:length(feedback.index.start_index)
    ex_start = feedback.index.start_index(i_trial);
    ex_end = feedback.index.end_index(i_trial);
    
    ex_log = []; ex_time = [];
    while(true)
        ex = singleClassification(decoder, feedback.data(ex_start:ex_start+window_length, decoder.eegChannels));
        ex_time = cat(1, ex_time, ex_start+window_length);
        ex_log = cat(1, ex_log, ex);
        ex_start = ex_start + decoder.fsamp/32;
        if (ex_start + window_length > ex_end)
            break;
        end
    end
    simulation_log{i_trial} = [ex_time, ex_log];
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions used in this script %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function index = computeIndex(trigger)
index = find(trigger == 10);
end