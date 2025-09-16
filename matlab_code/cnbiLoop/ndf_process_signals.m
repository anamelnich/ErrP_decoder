function ndf_process_signals(params)

global dataStream

key_idx = find(diff(dataStream.trigger));
trigger_values = trigger_data(key_idx + 1);

if (trigger_values(end) == goValue)
    dataStream.doProcess == true;
    dataStream.targetSize = length(trigger_data) + params.fsamp;
end

if (dataStream.doProcess == true && currentSize >= dataStream.targetSize)
    dataStream.doProcess = false;
    angleThreshold = process_signals(txtID, dataStream, dataStream, params);
    sendTiD(angleThreshold)
    initializeDataStream()
end

end 

function angleThreshold = process_signals(txtID, eeg, trigger, params)

behaviorSignals = get_behavior(txtID)
angleProfile = compute_angle(behaviorSignals)

eegOutput = process_eeg(eeg, trigger, params);

angleThreshold = compute_threshold(angleProfile, eegOutput)
end
