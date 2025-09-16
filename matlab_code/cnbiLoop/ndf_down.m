function ndf_down(decoder)

global ID ndf ids idm

disp('[ndf] Cleanup function: Finishing the loop');

pause(5);
% Tear down loop structure
tid_detach(ID);
ndf_close(ndf.sink);
idserializerrapid_delete(ids);
idmessage_delete(idm);
tid_delete(ID);

% computeModel(decoder.subjectID, false);

exit;
end