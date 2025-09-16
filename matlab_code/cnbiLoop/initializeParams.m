function decoder = initializeParams(decoder)

global stream ndf

stream.fsamp = ndf.conf.sf;
stream.frame_size = ndf.conf.samples; %32
stream.cycle_freq = round(stream.fsamp / stream.frame_size);
stream.cycle_time = (stream.frame_size / stream.fsamp);
stream.num_channels = length([decoder.eegChannels decoder.eogChannels]);

stream.flags.initFilter = true;

% max_sample = 1.0*stream.fsamp;
max_sample = 1.5*stream.fsamp;

signalLength = ceil(max_sample/stream.frame_size)*stream.frame_size;
singleClassification(decoder, rand(signalLength, length(decoder.eegChannels)));
stream.eeg = nan(signalLength, stream.num_channels);
stream.trigger = nan(signalLength, 1);
