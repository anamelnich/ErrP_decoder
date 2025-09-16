function [leftIdx, rightIdx] = findLeftRight(chanLabels)
% findLeftRight  Find matching left/right electrode index pairs
%
%   [leftIdx, rightIdx] = findLeftRight(chanLabels)
%
%   Inputs:
%     chanLabels : 1×n cell-array of channel names (e.g. {'F7','F3','FZ',…})
%
%   Outputs:
%     leftIdx  : m×1 vector of indices into chanLabels for the “left” (odd) sites
%     rightIdx : m×1 vector of indices into chanLabels for the corresponding “right” (even) sites
%
%   It assumes labels ending in 1,3,5,7 are left, and those ending in 2,4,6,8 are their mates.

  % Convert to string array
  labs = string(chanLabels);

  % Extract last character of each label
  lastChar = extractBetween(labs, strlength(labs), strlength(labs));
  charCell = cellstr(lastChar);  
  nums     = str2double(charCell);        % NaN for labels ending in 'Z'

  % Find odd-numbered (left) channels
  oddMask = ~isnan(nums) & mod(nums,2)==1;
  oddIdx  = find(oddMask);

  % Preallocate
  leftIdx  = [];
  rightIdx = [];

  % Loop through odd sites, look up their even partner
  for ii = 1:numel(oddIdx)
    iChan = oddIdx(ii);

    % base name is everything but the final digit
    base = extractBetween(labs(iChan), 1, strlength(labs(iChan))-1);

    % construct the expected even label
    evenLab = base + string(nums(iChan)+1);

    % find it in the full list
    ridx = find(labs == evenLab, 1);
    if ~isempty(ridx)
      leftIdx(end+1,1)  = iChan;
      rightIdx(end+1,1) = ridx;
    end
  end

end
