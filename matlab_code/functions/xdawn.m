function [filters, patterns, evokeds] = xdawn(epochs_data, y, n_components, epochSamples, reg, signal_cov, events, tmin, sfreq)
%XDawn Spatial filtering for ERP enhancement with optional sample-based window
%  [filters, patterns, evokeds] = xdawn(
%      epochs_data, y, n_components, epochSamples, reg, signal_cov, events, tmin, sfreq)
%
%  epochs_data   - [n_epochs x n_channels x n_times] epochs data
%  y             - [n_epochs x 1] class labels
%  n_components  - number of spatial components per class
%  epochSamples  - [1 x n_win] vector of time-sample indices to window evoked covariance
%                   or [] for full-epoch window
%  reg           - (unused) placeholder for regularization
%  signal_cov    - [n_channels x n_channels] precomputed signal covariance
%                   or [] to estimate from data
%  events        - [n_epochs x 3] events array (onset, ~, class)
%                   or [] for simple averaging
%  tmin          - start time relative to event (sec) [for overlap correction]
%  sfreq         - sampling frequency (Hz)
%
%  Outputs:
%  filters       - [(n_components*n_classes) x n_channels] spatial filters
%  patterns      - [(n_components*n_classes) x n_channels] spatial patterns
%  evokeds       - {n_classes x 1} cell of [n_channels x n_times] class-evoked

% Set defaults
if nargin < 4, epochSamples = []; end
if nargin < 5, reg          = []; end
if nargin < 6, signal_cov   = []; end
if nargin < 7, events       = []; end
if nargin < 8, tmin         = 0;  end
if nargin < 9, sfreq        = 1;  end

epochs_data = permute(epochs_data, [3 2 1]);

% Dimensions and classes
[n_epochs, n_channels, n_times] = size(epochs_data);
classes = unique(y);
n_classes = numel(classes);

% Determine evoked-sample window
if isempty(epochSamples)
    evo_samples = 1:n_times;
else
    % ensure indices within valid range
    evo_samples = epochSamples(epochSamples >= 1 & epochSamples <= n_times);
end

% Compute prototype evoked responses
if ~isempty(events)
    [evokeds_all, toeplitzs] = least_square_evoked(epochs_data, events, tmin, sfreq);
else
    evokeds_all = cell(n_classes,1);
    toeplitzs   = cell(n_classes,1);
    for i = 1:n_classes
        cls = classes(i);
        evokeds_all{i} = squeeze(mean(epochs_data(y==cls, :, :), 1));
        toeplitzs{i}   = 1;
    end
end

% Estimate full-epoch signal covariance if not provided
if isempty(signal_cov)
    data_concat = reshape(epochs_data, n_epochs*n_times, n_channels);
    signal_cov  = cov(data_concat);
end

% Initialize outputs
filters  = [];
patterns = [];
evokeds  = cell(n_classes,1);

% Loop per class to compute spatial filters
for i = 1:n_classes
    % Full-epoch prototype
    evo          = evokeds_all{i};           % [n_channels x n_times]
    toeplitz_mat = toeplitzs{i};             % scalar or matrix
    if isscalar(toeplitz_mat)
        evo_full = evo;
    else
        evo_full = evo * toeplitz_mat;
    end
    % Windowed samples
    evo_win = evo_full(:, evo_samples);

    % Evoked covariance
    evo_cov = cov(evo_win');  % [n_channels x n_channels]

    % Generalized eigendecomposition
    [V, D] = eig(evo_cov, signal_cov);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    V = V ./ sqrt(sum(V.^2, 1));

    % Spatial filters & patterns
    W = V(:, 1:n_components)';    % [n_components x n_channels]
    P = pinv(W);                   % [n_channels x n_components]

    filters  = [filters;  W];      % accumulate
    patterns = [patterns; P];      % accumulate
    evokeds{i} = evo;              % full-epoch evoked
end
end

%% Subfunction: least-squares evoked
function [evokeds, toeplitzs] = least_square_evoked(epochs_data, events, tmin, sfreq)
%LS_EVOKED Least-squares estimation of class-evoked responses
[n_epochs, n_channels, n_times] = size(epochs_data);
tmax = tmin + n_times/sfreq;

% Align events for overlap correction
events2 = events;
events2(:,1) = events(:,1) - events(1,1) - round(tmin*sfreq);
raw = construct_signal_from_epochs(epochs_data, events2, sfreq, tmin);

% Toeplitz parameters\ n_min = round(tmin*sfreq);
n_max = round(tmax*sfreq);
window = n_max - n_min;
n_samples = size(raw,2);
classes = unique(events(:,3));

evokeds  = cell(numel(classes),1);
toeplitzs = cell(numel(classes),1);
for ii = 1:numel(classes)
    sel = events(:,3)==classes(ii);
    trig = zeros(1, n_samples);
    trig(events(sel,1)+n_min) = 1;
    toeplitzs{ii} = toeplitz(trig(1:window), trig);
end

% Solve for evoked responses
X = vertcat(toeplitzs{:});                % [n_class*window x n_samples]
pred = (X*X') \ X;
E    = pred * raw';                        % [n_class*window x n_channels]
E    = E';                                % [n_channels x n_class*window]

evokeds = cell(numel(classes),1);
cnt = 1;
for ii = 1:numel(classes)
    evokeds{ii} = reshape(E(:, cnt:cnt+window-1), n_channels, window);
    cnt = cnt + window;
end
end

%% Subfunction: construct continuous signal
function raw = construct_signal_from_epochs(epochs, events, sfreq, tmin)
%RECONSTRUCT_SIGNAL Rebuild pseudo-continuous data from epochs
[n_epochs, n_channels, n_times] = size(epochs);
tmax = tmin + n_times/sfreq;

start = min(events(:,1)) + round(tmin*sfreq);
stop  = max(events(:,1)) + round(tmax*sfreq) + 1;
n_samples = stop - start;
raw = zeros(n_channels, nn_samples);

events_pos = events(:,1) - events(1,1);
for ei = 1:n_epochs
    onset = events_pos(ei);
    raw(:, onset+(1:n_times)) = squeeze(epochs(ei,:,:));
end
end

