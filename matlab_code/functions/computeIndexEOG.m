
function index = computeIndexEOG(trigger)
        [pos, typ] = ismember(trigger, [100 101 102 103]); % top right bottom left
        index.pos = find(pos); % Get positions of all distractor or no-distractor triggers
        typ_matched = typ(pos);
        index.typ = zeros(size(typ_matched));
        index.typ(typ_matched == 1) = 1; %dright
        index.typ(typ_matched == 2) = 2; %dleft
        index.typ(typ_matched == 3) = 3; %dleft
        index.typ(typ_matched == 4) = 4; %dleft
end
