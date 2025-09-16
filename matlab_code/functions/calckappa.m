function [kappa, expected_acc] = calckappa(labelTrue, labelPred)

    % Ensure labels are doubles (not logical or categorical)
    if ~isa(labelPred, 'double')
        labelPred = double(labelPred);
    end
    if ~isa(labelTrue, 'double')
        labelTrue = double(labelTrue);
    end

    % Compute confusion matrix
    cm = confusionmat(labelTrue, labelPred);

    % Make sure confusion matrix is 2x2
    if numel(cm) ~= 4
        error('Confusion matrix must be 2x2 (binary classification).');
    end

    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TP = cm(2,2);

    total = sum(cm(:));

    % Observed accuracy
    observed_acc = (TP + TN) / total;

    % Marginal frequencies
    row_marginals = sum(cm, 2);  % True class counts
    col_marginals = sum(cm, 1);  % Predicted class counts

    expected_acc = (row_marginals(1) * col_marginals(1) + ...
                    row_marginals(2) * col_marginals(2)) / total^2;

    % Cohen's Kappa
    kappa = (observed_acc - expected_acc) / (1 - expected_acc);

end