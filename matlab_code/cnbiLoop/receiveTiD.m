function output = receiveTiD()

global ID ids idm ndf

if (tid_isattached(ID) == false)
    tid_attach(ID);
end

if (tid_isattached(ID) == true)
    if (tid_getmessage(ID, ids) == true)
        output = idmessage_getevent(idm);
    else
        output = [];
    end
end