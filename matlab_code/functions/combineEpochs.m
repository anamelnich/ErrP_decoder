function combinedEpochs = combineEpochs(epochStructs)
% combineEpochs concatenates multiple epoch structures with RT
%
%   combinedEpochs = combineEpochs(epochStructs)
%   Inputs:
%     epochStructs - cell array of structs, each must have fields:
%         .data    [time x channels x trials]
%         .labels  [trials x 1]
%         .file_id [trials x 1]
%         .eof     [scalar or vector end-of-file indices]
%         .RT      [trials x 1]  (optional: reaction times)
%   Output:
%     combinedEpochs struct with fields:
%         .data, .labels, .file_id, .eof, .RT

% initialize
combinedEpochs = struct();
combinedEpochs.data    = [];
combinedEpochs.labels  = [];
combinedEpochs.file_id = [];
combinedEpochs.eof     = [];
combinedHasRT = true;
combinedEpochs.RT      = [];
combinedEpochs.tpos    = [];
% combinedEpochs.dpos    = [];

file_id_offset = 0;

t = numel(epochStructs);
for i = 1:t
    cur = epochStructs{i};
    nTrials = numel(cur.labels);
    %----- concatenate data -----
    if isempty(combinedEpochs.data)
        combinedEpochs.data = cur.data;
    else
        combinedEpochs.data = cat(3, combinedEpochs.data, cur.data);
    end
    % labels
    combinedEpochs.labels = [combinedEpochs.labels; cur.labels];
    % file_id (adjusted)
    adjFileID = cur.file_id + file_id_offset;
    combinedEpochs.file_id = [combinedEpochs.file_id; adjFileID];
    % update offset
    if ~isempty(cur.file_id)
        file_id_offset = file_id_offset + max(cur.file_id);
    end
    % eof
    combinedEpochs.eof = [combinedEpochs.eof; cur.eof(:)];

    % RT handling
    if isfield(cur, 'RT') && numel(cur.RT)==nTrials
        combinedEpochs.RT = [combinedEpochs.RT; cur.RT(:)];
        combinedEpochs.tpos = [combinedEpochs.tpos; cur.tpos(:)];
        % combinedEpochs.dpos = [combinedEpochs.dpos; cur.dpos(:)];
    else
        % fill with NaNs for missing or mismatched RT
        combinedHasRT = false;
        combinedEpochs.RT = [combinedEpochs.RT; nan(nTrials,1)];
    end
end

% If any epoch lacked valid RT, warn once
if ~combinedHasRT
    warning('combineEpochs:MissingRT', 'Some epochs missing RT or length mismatch; filled with NaNs.');
end
end

