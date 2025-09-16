function idx = fisher(epochs,labels,params)

% erpEpochs: T×C×N, trainLabels: N×1 with values 0 or 1
epochs = epochs(params.resample.time,:,:);

chanFeat = squeeze( mean(epochs, 1) )';   % trial x channel

mu0    = mean(chanFeat(labels==0,:), 1); % 1 x chan number 
mu1    = mean(chanFeat(labels==1,:), 1);
var0   = var( chanFeat(labels==0,:), 0, 1);
var1   = var( chanFeat(labels==1,:), 0, 1);

Fscore = (mu1 - mu0).^2 ./ (var0 + var1);   % 1× chan number 

[~, order] = sort(Fscore, 'descend');
idx      = order(1:10);

