import torch
from laplace import Laplace
import numpy as np
from typing import Tuple, Dict

def build_laplace(
    model: torch.nn.Module,
    likelihood: str,
    subset_of_weights: str,
    hessian_structure: str,
    temperature: float = 1.0,
    feature_reduction: str = "pick_last",
    backend=None,
    n_subset: int = None,
):
    """
    Construct a Laplace instance. 
    Pass backend=AsdlGGN (class) if needed; pass n_subset for GP.
    """
    kwargs = dict(
        model=model,
        likelihood=likelihood,
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        temperature=temperature,
    )
    if feature_reduction == "pick_last":
        kwargs["feature_reduction"] = feature_reduction
    if backend is not None:
        kwargs["backend"] = backend
    if n_subset is not None:
        kwargs["n_subset"] = n_subset
    return Laplace(**kwargs)

def fit_laplace(la, train_loader, val_loader, optimize_prior: bool = True):
    la.fit(train_loader=train_loader)
    if optimize_prior:
        la.optimize_prior_precision()
    return la

def laplace_inference(
    model: torch.nn.Module,
    la: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the base model and Laplace posterior on the entire dataloader.

    Returns:
      logits      : np.ndarray of shape [N, C]
      la_probs    : np.ndarray of shape [N, C]  (Laplace predictive probs)
      labels      : np.ndarray of shape [N, C]  (true multi-hot)
      base_probs  : np.ndarray of shape [N, C]  (sigmoid(logits))
    """
    all_la_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # split inputs vs labels
            data = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["labels"].float().to(device)
            # Laplace predictive probs (LA)
            la_p = la(batch, pred_type = 'nn', link_approx = 'mc', n_samples=n_samples)                             # [B, C]

            all_la_probs.append(la_p.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_la_probs = np.vstack(all_la_probs)
    all_labels   = np.vstack(all_labels)


    return all_la_probs, all_labels