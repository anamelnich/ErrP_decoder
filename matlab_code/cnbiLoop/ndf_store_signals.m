function ndf_store_signals(eeg, trigger, decoder)

global stream
%Shift the EEG and trigger buffers to make room for new data
stream.eeg(1:(end-stream.frame_size), :) = stream.eeg((stream.frame_size+1):end,:);
stream.trigger(1:(end-stream.frame_size), :) = stream.trigger((stream.frame_size+1):end,:);

stream = bandpass_filter(eeg, stream, decoder);
stream.trigger((end-stream.frame_size+1):end, :) = trigger;

function stream = bandpass_filter(eeg, stream, decoder)

if (stream.flags.initFilter)
    stream.flags.initFilter = false;
    stream.filter.Zf = [];
    for i_ch = 1:stream.num_channels
        [stream.eeg((end-stream.frame_size+1):end, i_ch), stream.filter.Zf(:,i_ch)] = ...
            filter(decoder.spectralFilter.b, decoder.spectralFilter.a, eeg(:, i_ch));
    end
else
    for i_ch = 1:stream.num_channels
        [stream.eeg((end-stream.frame_size+1):end, i_ch), stream.filter.Zf(:,i_ch)] = ...
            filter(decoder.spectralFilter.b, decoder.spectralFilter.a, eeg(:, i_ch), stream.filter.Zf(:, i_ch));
    end
end
%commented out cause don't have EOG filter
% stream.eeg((end-stream.frame_size+1):end, decoder.eegChannels) = ...
%     stream.eeg((end-stream.frame_size+1):end, decoder.eegChannels) - ...
%     stream.eeg((end-stream.frame_size+1):end, decoder.eogChannels) * decoder.eogFilter;
