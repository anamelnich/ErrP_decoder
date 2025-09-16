function index = compute_index(trigger, trigger_codes)
% COMPUTE_INDEX  For each move, label epoch as 0 (correct) or 1 (error).
% Inputs:
%   trigger        - 1D array of event codes over time/samples
%   trigger_codes  - [move_code, neg_code], e.g., [100 104]
% Output:
%   index.pos  - sample indices of move events (move_code)
%   index.typ  - labels per move: 0 = correct, 1 = negative (104 occurred before next move)

move_code = trigger_codes(1);
neg_code  = trigger_codes(2);

% Keep only relevant events, preserving order
is_rel   = (trigger == move_code) | (trigger == neg_code);
rel_vals = trigger(is_rel);
rel_pos  = find(is_rel);

% Indices of moves among relevant events
k_move = find(rel_vals == move_code);

% For each move, check if the next relevant event is a neg_code (104)
has_next      = k_move < numel(rel_vals);
is_neg_next   = false(size(k_move));
is_neg_next(has_next) = (rel_vals(k_move(has_next) + 1) == neg_code);

% Build outputs
index.pos = rel_pos(k_move);           % positions of moves in the original trigger vector
index.typ = double(is_neg_next);       % 0 = correct, 1 = negative
end


% function index = compute_index(trigger, trigger_codes)
% % COMPUTE_INDEX  Find move triggers and label as neutral (0) or negative (2).
% % Inputs:
% %   trigger        - 1D array of trigger values
% %   trigger_codes  - vector with [move_code, neg_code]
% % Outputs:
% %   pos - positions (indices) of move triggers
% %   typ - labels: 0=neutral, 1=negative
% 
% % Identify relevant triggers
% is_relevant = ismember(trigger, trigger_codes);
% index_pos   = find(is_relevant);
% code_types  = trigger(is_relevant);
% 
% % Initialize types: 0 = neutral, 2 = negative
% index_typ = zeros(size(code_types));
% index_typ(code_types == trigger_codes(2)) = 1;
% 
% % Assign negative label to preceding move
% for i = 1:(numel(index_typ)-1)
%     if index_typ(i) == 0 && index_typ(i+1) == 1
%         index_typ(i) = 1;
%     end
% end
% 
% % Keep only moves
% move_mask = (code_types == trigger_codes(1));
% index.pos = index_pos(move_mask);
% index.typ = index_typ(move_mask);
% end
