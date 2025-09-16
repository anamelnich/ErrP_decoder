
function index = computeIndex(trigger,trigType)

    if trigType == 1
        [pos, typ] = ismember(trigger, [102 104 100 110]); %110, subjetcts 10-15 and above
    elseif trigType == 0
        [pos, typ] = ismember(trigger, [202 204 100 110]); %110
    elseif trigType == 2
        [pos, typ] = ismember(trigger, [106 107 108 100 110]); % left distractors only with 8 shapes
    elseif trigType == 3
        [pos, typ] = ismember(trigger, [204 100 110]); % left distractors only
    elseif trigType == 4
        [pos, typ] = ismember(trigger, [104 100 110]); % left distractors only
    end 
    index.pos = find(pos); 
    typ_matched = typ(pos);
    index.typ = zeros(size(typ_matched));
    if trigType == 2
        index.typ(typ_matched < 4) = 1; %dleft
        index.typ(typ_matched >= 4) = 0; %dnone
    elseif trigType == 3 || trigType == 4
        index.typ(typ_matched == 1) = 1; %dleft
        index.typ(typ_matched > 1) = 0; %dnone
    else
        index.typ(typ_matched == 1) = 1; %dright
        index.typ(typ_matched == 2) = 2; %dleft
        index.typ(typ_matched > 2) = 0; %dnone
    end

end

