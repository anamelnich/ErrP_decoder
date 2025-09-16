function bandPower = compute_power(data, params)
% Extract power from a specific frequency band using CWT (Morlet)
%
% INPUTS:
%   data                 - [time x channels x epochs] EEG data
%   params.fsamp         - Sampling frequency in Hz
%   params.tfr.freqBand  - 1x2 vector [lowFreq highFreq], e.g., [4 7] for theta
%
% OUTPUT:
%   bandPower - [time x channels x epochs] power in the specified band

[nt, nchan, nepochs] = size(data);
bandPower = zeros(nt, nchan, nepochs);

for ch = 1:nchan
    for ep = 1:nepochs
        % Continuous wavelet transform using Morlet wavelets
        [cfs, freqs] = cwt(data(:, ch, ep), 'amor', params.fsamp);  % cfs: [freq x time]
        freq_idx = freqs >= params.tfr.freqBand(1) & freqs <= params.tfr.freqBand(2);
        power_band = abs(cfs(freq_idx, :)).^2;
        bandPower(:, ch, ep) = mean(power_band, 1)';
    end
end
end
