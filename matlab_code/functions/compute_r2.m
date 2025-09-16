function r2 = compute_r2(data, labels)

data_cell = mat2cell3(data);

r2 = cellfun(@(x) corr(squeeze(x),labels, 'type', 'Pearson').^2, data_cell, 'UniformOutput', 0);

r2 = cell2mat(r2);

function output = mat2cell3(input) %input 92x1x240
output = mat2cell(input, ones(1,size(input,1)), ones(1,size(input,2)), size(input,3)); %92x1, where each cell is 1x1x240