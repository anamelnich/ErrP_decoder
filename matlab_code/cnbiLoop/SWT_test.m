function ndf_main()

global stream ndf ID ids idm

% warning('off', 'all');
% Include any required toolboxes
ndf_include(); %adds paths to CNBI toolkit and eegc3

try
   
    tid_attach(ID);
    

    %% Main Loop %%
    for k = 0:100
        send_tid(k);           
        pause(0.02);           
    end

catch exception
    ndf_printexception(exception);
end
end