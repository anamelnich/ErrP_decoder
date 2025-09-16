function ndf_initialization()

    global ID ndf ids idm

	% Initialize loop structure
	cl = cl_new();

	% Connect to the CnbiTk loop
	if(cl_connect(cl) == false)
		disp('[ndf] Cannot connect to CNBI Loop, killing matlab');
		exit;
	end	

	% Prepare NDF srtructure
	ndf.conf  = {};
	ndf.size  = 0;
	ndf.frame = ndf_frame();
	ndf.sink  = ndf_sink('/tmp/cl.pipe.ndf.0'); % Connect to /pipe0

    % Create a single ID client for sending and receiving events to/from a feedback
    % (or other loop modules). The possibility for separate ID clients for
    % receiving/sendind exists
    
    ID = tid_new(); % Create ID client
    idm = idmessage_new(); % Create ID message for both sening/receiving
    ids = idserializerrapid_new(idm); % Create ID message serializer
    % Configure ID message
    idmessage_setdescription(idm, 'io');
    idmessage_setfamilytype(idm, idmessage_familytype('biosig'));
    idmessage_dumpmessage(idm);
    ida = '/bus'; % Alias of iD bus
    
	% Pipe opening and NDF configuration
	% - Here the pipe is opened
	% - ... and the NDF ACK frame is received
	disp('[ndf] Receiving ACK...');
	[ndf.conf, ndf.size] = ndf_ack(ndf.sink);

end