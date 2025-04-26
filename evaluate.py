# evaluate.py

import torch
import numpy as np
from typing import Tuple, Dict

from metrics import (
    elementwise_accuracy,
    subset_accuracy,
    hamming_loss,
    f1_scores,
    roc_auc_scores,
    log_loss_multilabel,
    brier_score_multilabel,
    get_calibration,
)

def collect_outputs(
    model: torch.nn.Module,
    la: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the base model and Laplace posterior on the entire dataloader.

    Returns:
      logits      : np.ndarray of shape [N, C]
      la_probs    : np.ndarray of shape [N, C]  (Laplace predictive probs)
      labels      : np.ndarray of shape [N, C]  (true multi-hot)
      base_probs  : np.ndarray of shape [N, C]  (sigmoid(logits))
    """
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # split inputs vs labels
            data = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["labels"].float().to(device)

            # base model logits (MAP)
            logits = model(data)                         # [B, C]

            all_logits.append(logits.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

    all_logits   = np.vstack(all_logits)

    all_labels   = np.vstack(all_labels)
    base_probs   = torch.sigmoid(torch.tensor(all_logits)).numpy()

    return all_logits, base_probs, all_labels


def compute_all_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Given predicted probabilities, binary preds, and true labels,
    compute a dictionary of common multi-label metrics.
    """
    metrics = {}
    metrics["accuracy"]     = subset_accuracy(labels, preds)
    metrics["f1_micro"]       = f1_scores(labels, preds, average="micro")
    metrics["f1_macro"]       = f1_scores(labels, preds, average="macro")
    metrics["roc_auc_micro"]  = roc_auc_scores(labels, probs, average="micro")
    metrics["roc_auc_macro"]  = roc_auc_scores(labels, probs, average="macro")
    metrics["log_loss"]       = log_loss_multilabel(labels, probs)
    metrics["brier_score"]    = brier_score_multilabel(labels, probs)

    ece_per_class, ece_global, mce_global = get_calibration(probs, labels)
    metrics["ece_global"] = ece_global
    metrics["mce_global"] = mce_global
    # optionally include mean per-class ECE
    metrics["ece_mean_class"] = float(np.mean(ece_per_class))

    return metrics


def evaluate(
    model: torch.nn.Module,
    la: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    End-to-end evaluation that returns two metric dicts:
      - base_metrics : using the GPT-2 head
      - laplace_metrics : using the Laplace posterior
    """
    logits, base_probs, la_probs, labels = collect_outputs(model, la, dataloader, device)
    base_preds = (base_probs >= 0.5).astype(int)
    la_preds   = (la_probs   >= 0.5).astype(int)

    base_metrics   = compute_all_metrics(base_probs, base_preds, labels)
    laplace_metrics = compute_all_metrics(la_probs, la_preds, labels)

    return base_metrics, laplace_metrics