% function [trainingData, bestItrData] = computeModel(subjectID)
function computeModel(subjectID)

%% ====================== Initialization ====================== %%
clearvars -except subjectID cfg S;
close all; rng('default');
addpath(genpath('../functions'));

%% ======================== Load Data ========================= %%

dataPath = [pwd '/../../data/' subjectID];
data = loadData(dataPath);
% data = loadDataDecoding(dataPath);

%% ============== Set Params and Preprocess Data ============== %%
cfg = setParams(data.header);
data.index = compute_index(data.data(:,cfg.triggerChannel),[100,104]);

%% ==================== Bandpass Filter ======================= %%
[b, a] = butter(cfg.spectralFilter.order, cfg.spectralFilter.freqs./(cfg.fsamp/2), 'bandpass');
cfg.spectralFilter.b = b;
cfg.spectralFilter.a = a;
data.data = filter(b, a, data.data);


%% ======================== Epoching ========================== %%

d = data;

epochs.data = nan(length(cfg.epochSamples), length(cfg.chanLabels), length(d.index.pos));
epochs.labels = d.index.typ;
epochs.file_id = nan(length(d.index.typ), 1);

for t = 1:length(d.index.pos)
    epochs.data(:, :, t) = d.data(d.index.pos(t) + cfg.epochSamples, :);
    epochs.file_id(t) = find(d.index.pos(t) <= d.eof, 1, 'first');
end
epochs.data = epochs.data(:,cfg.eegChannels,:);
data.epochs = epochs;
data.epochs.eof = d.eof;
%% ====================== Pruning & Model Performance ======================== %%

[performance, bestItrData] = iterativePrune(data.epochs, cfg, 20)
history = performance.history;
nIter = 20;
iters = 1:nIter;
plotPruningMetrics(iters, history.ACC, history.AUPRC, history.TPR, history.TNR, history.nTrials, subjectID);
plotERPpruned(data.epochs,bestItrData,cfg)

bestItrData.performance = performance;
%%
% save('subject2.mat','bestItrData');


%% ===================== Build Model ======================== %%
[decoder,~] = computeDecoder(bestItrData.data, bestItrData.labels,cfg);

decoder.eegChannels = cfg.eegChannels; 
decoder.eogChannels = cfg.eogChannels;
decoder.spectralFilter = cfg.spectralFilter;
decoder.threshold = performance.threshold;
% decoder.threshold = 0.45;
% decoder.resample.time = decoder.resample.time - decoder.epochOnset;
decoder.performance = performance;
decoder.subjectID = subjectID;
decoder.datetime = datetime;
decoder.onlinePosteriors = [];
disp(' ');
disp('Decoder Updated at');
disp(decoder.datetime);

save(sprintf('./decoders/%s_decoder.mat', subjectID), 'decoder');
save('../cnbiLoop/decoder.mat', 'decoder');

end
% 
% ================= Helper Function ================= %%
function [tpr,tnr,acc] = printConfusionMatrix(trueLabels, predictedLabels)
cm = confusionmat(logical(trueLabels), predictedLabels);
disp('Confusion Matrix (with labels):');
disp('--------------------------------');
disp('            Pred=0    Pred=1');
fprintf('True=0:       %3d       %3d\n', cm(1,1), cm(1,2));
fprintf('True=1:       %3d       %3d\n', cm(2,1), cm(2,2));
tnr = cm(1,1) / sum(cm(1,:));
tpr = cm(2,2) / sum(cm(2,:));
acc = sum(diag(cm)) / sum(cm(:));
fprintf('TNR: %.2f | TPR: %.2f | Accuracy: %.2f\n\n', tnr, tpr, acc);
end

