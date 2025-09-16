function sendTiD(value)

global ID ids idm ndf

if (tid_isattached(ID) == false)
    tid_attach(ID);
end
        
idmessage_setevent(idm, value);
tid_setmessage(ID, ids, ndf.frame.index);

