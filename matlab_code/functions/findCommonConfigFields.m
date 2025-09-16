function [commonFields, commonValues] = findCommonConfigFields(T, idx)
% findCommonCfgFieldsInTable  Find fields (including nested) common across configs
%
%   findCommonCfgFieldsInTable(T)  
%     prints all fields (nested) that have the same value in every row of T.
%
%   findCommonCfgFieldsInTable(T, idx)  
%     restricts to the rows T(idx,:), then prints.
%
%   [commonFields, commonValues] = findCommonCfgFieldsInTable(...)  
%     returns the results instead of printing:
%       commonFields – K×1 cell of dot-notation field names
%       commonValues – K×1 cell of their common values
%
% Example:
%   % show common fields in rows [1 3 5] of your table:
%   findCommonCfgFieldsInTable(T, [1 3 5]);
%
%   % get them programmatically:
%   [cf, cv] = findCommonCfgFieldsInTable(T);

    % default to all rows
    if nargin<2 || isempty(idx)
        idx = 1:height(T);
    end

    % unpack into a struct array
    % assume T.cfg is a cell of 1×1 structs
    configs = vertcat( T.cfg(idx) );

    % run the recursive search
    [commonFields, commonValues] = recurse(configs, "");

    % if no outputs, print them
    if nargout==0
        fprintf('Common fields across configs %s:\n', mat2str(idx));
        for k = 1:numel(commonFields)
            val = commonValues{k};
            fld = commonFields{k};
            if isnumeric(val) || islogical(val) || ischar(val) || isstring(val)
                txt = mat2str(val);
                fprintf('  %s = %s\n', fld, txt);
            elseif isstruct(val)
                subs = fieldnames(val);
                fprintf('  %s is struct with subfields: %s\n', fld, strjoin(subs, ', '));
            else
                fprintf('  %s (class %s)\n', fld, class(val));
            end
        end
    end

    %% nested recursive function
    function [cf, cv] = recurse(cfgs, prefix)
        cf = {};
        cv = {};
        fn = fieldnames(cfgs);
        for i = 1:numel(fn)
            fld = fn{i};
            % collect this field from all configs
            vals = arrayfun(@(s)s.(fld), cfgs, 'UniformOutput', false);
            % check equality
            if all(cellfun(@(v) isequaln(v, vals{1}), vals(2:end)))
                fullName = char(prefix + fld);
                % if it's a struct, recurse deeper
                if isstruct(vals{1})
                    subCfgs = vertcat(vals{:});
                    [subF, subV] = recurse(subCfgs, prefix + fld + ".");
                    cf = [cf; subF];
                    cv = [cv; subV];
                else
                    cf{end+1,1} = fullName;
                    cv{end+1,1} = vals{1};
                end
            end
        end
    end
end

