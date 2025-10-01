function ndf_main()

global stream ndf ID ids idm

% warning('off', 'all');
% Include any required toolboxes
ndf_include(); %adds paths to CNBI toolkit and eegc3
addpath(genpath('../decoder'));
addpath(genpath('../functions'));


skip_iterations = true;
% Prepare and enter main loop
try
    load('./decoder.mat');
    disp('Decoder Updated at');
    disp(decoder.datetime);

    ndf_initialization(); %sets up ndf configuration, should automatically setup ndf with 64 ch based on incoming data
    decoder = initializeParams(decoder);
    cleanupObj = onCleanup(@() ndf_down(decoder));

    tid_attach(ID);
    disp('[ndf] Receiving NDF frames...');

    %%
    update_flag = false;
    %% Main Loop %%
    while(true)
        tic
        [ndf.frame, ndf.size] = ndf_read(ndf.sink, ndf.conf, ndf.frame); %read data, outputs frame = data and count = data size
        time_frame = 1000*toc;

        % Acquisition is down, exit
        if(ndf.size == 0)
            disp('[ndf] Broken pipe');
            break;
        end

        if ((time_frame < 20) && (skip_iterations)) %skip iteration if reading data takes less than 20 msec 
            disp(['Skipping iteration. Time was: ' num2str(time_frame) ' ms']);
        else
            skip_iterations = false;

            eeg_input = ndf.frame.eeg; %(samples x eeg_channels)
            eog_input = ndf.frame.exg; %(samples x exg_channels)
            trigger_input = ndf.frame.tri; %(samples x tri_channels)

            %store EEG and trigger data in stream, includes bandpass filter ...
            % based on spatial filter in decoder, also has EOG filter (commented out)
            ndf_store_signals([eeg_input, eog_input], trigger_input, decoder); 

            if (~any(isnan(stream.eeg(:))))
                %returns sample (out of 768) where one of these triggers is found
                first_index = find(ismember(stream.trigger, [100]), 1, 'first'); % look for move trigger
                %disp(first_index)
                if (first_index >= 256) & (first_index <= 308) % 0.5 sec baseline, need 256 for decoder.baseline_idx to work correctly
                    label_value = stream.trigger(first_index);
                    win = first_index + decoder.cfg.epochSamples;
%                     fprintf('Label value at first_index (%d): %d\n', first_index, label_value);
                    [ex_posterior, ~] = singleClassification(decoder, stream.eeg(win, decoder.eegChannels));
%                     disp(['Time Frame: ' num2str(time_frame, '%.2f') ' Posteriors: ' num2str(ex_posterior, ' %.2f')]);
                    decoder.onlinePosteriors = [decoder.onlinePosteriors, ex_posterior];
                    stream.trigger(first_index) = 0;
%                     sendTiD(1 + (ex_posterior > decoder.threshold)); % sends 1 if below threshold, 2 if above
                    sendTiD(ex_posterior);

                end
            end

            if (receiveTiD() == 20)
                break;
            end
        end
    end
folderPath = './online_decoders';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = fullfile(folderPath,['decoder_' timestamp '.mat']);
save(filename, 'decoder');

catch exception
    ndf_printexception(exception);
end
end
