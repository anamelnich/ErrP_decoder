# evaluation.py: Evaluation metrics and utilities
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(labels, posteriors):
    """
    Compute ROC, find optimal threshold, and print performance metrics.

    Parameters
    ----------
    labels : array-like of shape (n_trials,)
        True binary labels (0 or 1).
    posteriors : array-like of shape (n_trials, n_classes)
        Posterior probabilities for all classes. Positive class (1) is at index 1.

    Returns
    -------
    results : dict with keys
        - 'fpr', 'tpr', 'thresholds', 'auc'
        - 'opt_threshold', 'opt_fpr', 'opt_tpr'
        - 'confusion_matrix', 'tnr', 'tpr_at_opt'
    """
    labels = np.asarray(labels)
    posteriors = np.asarray(posteriors)
    if posteriors.ndim != 2:
        raise ValueError(f"posteriors must be a 2D array of shape (n_trials, n_classes), got shape {posteriors.shape}")
    if posteriors.shape[1] < 2:
        raise ValueError(f"posteriors must have at least 2 columns (for binary classification), got shape {posteriors.shape}")
    if posteriors.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of trials in posteriors ({posteriors.shape[0]}) does not match number of labels ({labels.shape[0]})")
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Labels must be binary (0 or 1), got unique values: {unique_labels}")

    posteriors_pos = posteriors[:, 1] # positive class

    fpr, tpr, thr = roc_curve(labels, posteriors_pos, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # optimal threshold by maximizing Youden's J = TPR - FPR
    youden = tpr - fpr
    idx_opt = np.argmax(youden)
    opt_thr = thr[idx_opt]
    opt_fpr = fpr[idx_opt]
    opt_tpr = tpr[idx_opt]

    preds = (posteriors_pos >= opt_thr).astype(int)
    cm = confusion_matrix(labels, preds)
    # TNR = TN / (TN + FP)
    tnr_opt = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else np.nan
    # TPR = TP / (TP + FN)
    tpr_opt = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else np.nan

    print("== Synchronous Classification == ")
    print(f"AUC score : {roc_auc:.2f}   Threshold: {opt_thr:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"TNR: {tnr_opt:.2f}   TPR: {tpr_opt:.2f}")

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thr,
        'auc': roc_auc,
        'opt_threshold': opt_thr,
        'opt_fpr': opt_fpr,
        'opt_tpr': opt_tpr,
        'confusion_matrix': cm,
        'tnr': tnr_opt,
        'tpr': tpr_opt
    }

def plot_posteriors_by_class(labels, posteriors, class_names=None, bins=30):
    """
    Plot the distribution of positive class posteriors for each true class.

    Parameters
    ----------
    labels : array-like of shape (n_trials,)
        True binary labels (0 or 1).
    posteriors : array-like of shape (n_trials, n_classes)
        Posterior probabilities for all classes. Positive class (1) is at index 1.
    class_names : list or None
        Optional names for the classes, e.g., ['No Feedback', 'Negative Feedback']
    bins : int
        Number of bins for the histogram
    """
    labels = np.asarray(labels)
    posteriors = np.asarray(posteriors)
    if posteriors.ndim != 2 or posteriors.shape[1] < 2:
        raise ValueError("posteriors must be a 2D array with at least 2 columns.")
    if posteriors.shape[0] != labels.shape[0]:
        raise ValueError("Number of trials in posteriors and labels must match.")
    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    pos_probs = posteriors[:, 1]
    plt.figure(figsize=(8, 5))
    for c in [0, 1]:
        mask = labels == c
        if np.sum(mask) == 0:
            continue
        sns.histplot(pos_probs[mask], bins=bins, kde=True, stat='density', label=class_names[c], alpha=0.6)
    plt.xlabel('Posterior Probability (Class 1)')
    plt.ylabel('Density')
    plt.title('Posterior Distributions by True Class')
    plt.legend()
    plt.tight_layout()
    plt.show()


# def calckappa(labels, preds):
#     """Calculate Cohen's kappa."""
#     pass

