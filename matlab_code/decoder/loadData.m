% Imports GDF file 

function training = loadData(path)

gdfFiles = dir(fullfile(path, '*.gdf'));
fullPaths = fullfile({gdfFiles.folder}, {gdfFiles.name});
training = [];

for i = 1:length(fullPaths)
    fileName = fullPaths{i};
    [signal, header] = sload(fileName);
    if (isfield(training, 'data'))
        training.data = cat(1, training.data, signal);
        training.eof = cat(1, training.eof, size(training.data, 1));
    else
        training.data = signal;
        training.header = header;
        training.eof = size(signal, 1);
    end
end
delete sopen.mat
end