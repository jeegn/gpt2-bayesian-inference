# metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    hamming_loss as sk_hamming_loss,
    f1_score as sk_f1_score,
    roc_auc_score as sk_roc_auc_score,
    log_loss as sk_log_loss,
    brier_score_loss as sk_brier_score_loss,
)


def elementwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Element-wise accuracy (equivalent to 1 - Hamming loss).
    Flattens both arrays and computes overall match rate.
    """
    return sk_accuracy(y_true.flatten(), y_pred.flatten())


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exact-match (subset) accuracy: fraction of samples
    where *all* labels are predicted correctly.
    """
    return np.mean(np.all(y_true == y_pred, axis=1))


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hamming loss: fraction of labels incorrectly predicted.
    """
    return sk_hamming_loss(y_true, y_pred)


def f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "micro",
    zero_division: int = 0
) -> float:
    """
    F1 score (micro or macro) with zero_division handling.
    """
    return sk_f1_score(
        y_true, y_pred,
        average=average,
        zero_division=zero_division
    )


def roc_auc_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    average: str = "micro"
) -> float:
    """
    ROC AUC (micro or macro) over only the labels that vary.
    """
    # pick only columns where there is at least one positive and one negative
    varying = [
        i for i in range(y_true.shape[1])
        if 0 < y_true[:, i].sum() < y_true.shape[0]
    ]
    return sk_roc_auc_score(
        y_true[:, varying],
        y_prob[:, varying],
        average=average
    )


def log_loss_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Multi-label log loss (flattened binary cross-entropy).
    """
    return sk_log_loss(y_true.ravel(), y_prob.ravel())


def brier_score_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Multi-label Brier score (mean squared error on probabilities).
    """
    return sk_brier_score_loss(y_true.ravel(), y_prob.ravel())


def get_calibration(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 25
) -> tuple[np.ndarray, float, float]:
    """
    Compute calibration metrics for multi-label predictions.

    Args:
      y_prob: [N, C] array or tensor of predicted probabilities.
      y_true: [N, C] array or tensor of multi-hot true labels (0/1).
      n_bins: number of equal-width bins to use.

    Returns:
      ece_per_class: np.ndarray of length C, ECE for each class.
      ece_global:     float, global ECE weighted by bin size.
      mce_global:     float, maximum calibration error across all global bins.
    """

    N, C = y_prob.shape
    # Bin edges
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # 2) Per-class ECE
    ece_per_class = np.zeros(C, dtype=float)
    for c in range(C):
        conf_c = y_prob[:, c]
        true_c = y_true[:, c]
        ece_c = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i+1]
            mask = (conf_c > lo) & (conf_c <= hi)
            if not mask.any():
                continue
            acc_bin  = true_c[mask].mean()
            conf_bin = conf_c[mask].mean()
            ece_c   += np.abs(acc_bin - conf_bin) * (mask.sum() / N)
        ece_per_class[c] = ece_c

    # 3) Global ECE & MCE (flatten across all classes)
    conf_flat = y_prob.flatten()
    true_flat = y_true.flatten()
    ece_global = 0.0
    mce_global = 0.0
    total = conf_flat.shape[0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (conf_flat > lo) & (conf_flat <= hi)
        if not mask.any():
            continue
        acc_bin  = true_flat[mask].mean()
        conf_bin = conf_flat[mask].mean()
        gap      = np.abs(acc_bin - conf_bin)
        weight   = mask.sum() / total
        ece_global += gap * weight
        mce_global  = max(mce_global, gap)

    return ece_per_class, ece_global, mce_global

