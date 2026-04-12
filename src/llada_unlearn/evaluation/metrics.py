"""Evaluation metrics for unlearning experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from llada_unlearn.config import LossConfig
from llada_unlearn.models.base import ModelWrapper


@torch.no_grad()
def forget_quality(
    model: ModelWrapper,
    forget_loader: DataLoader,
    loss_config: LossConfig,
) -> dict:
    """Measure how well the model has forgotten.

    Higher loss on forget set = better forgetting.
    Returns mean loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for (batch,) in forget_loader:
        batch = batch.to(model.device)
        fwd = model.forward_pass(batch, loss_config)
        total_loss += fwd.loss.item()
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    return {"loss": mean_loss, "perplexity": np.exp(min(mean_loss, 100))}


@torch.no_grad()
def model_utility(
    model: ModelWrapper,
    test_loader: DataLoader,
    loss_config: LossConfig,
) -> dict:
    """Measure retained model utility on held-out test data.

    Lower loss = better utility preservation.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for (batch,) in test_loader:
        batch = batch.to(model.device)
        fwd = model.forward_pass(batch, loss_config)
        total_loss += fwd.loss.item()
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    return {"loss": mean_loss, "perplexity": np.exp(min(mean_loss, 100))}


@torch.no_grad()
def membership_inference(
    model: ModelWrapper,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    loss_config: LossConfig,
) -> dict:
    """Loss-based membership inference attack.

    Computes per-sample losses for members (forget set) and non-members,
    then reports AUC. Lower AUC (closer to 0.5) = better unlearning.
    """
    model.eval()

    def _collect_losses(loader: DataLoader) -> list[float]:
        losses = []
        for (batch,) in loader:
            batch = batch.to(model.device)
            for i in range(batch.shape[0]):
                fwd = model.forward_pass(batch[i : i + 1], loss_config)
                losses.append(fwd.loss.item())
        return losses

    member_losses = _collect_losses(member_loader)
    non_member_losses = _collect_losses(non_member_loader)

    # Labels: 1 = member, 0 = non-member
    labels = [1] * len(member_losses) + [0] * len(non_member_losses)
    # Scores: negative loss (lower loss = more likely member)
    scores = [-v for v in member_losses] + [-v for v in non_member_losses]

    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5

    return {
        "auc": auc,
        "member_mean_loss": np.mean(member_losses),
        "non_member_mean_loss": np.mean(non_member_losses),
    }


@torch.no_grad()
def kl_divergence(
    model_a: ModelWrapper,
    model_b: ModelWrapper,
    data_loader: DataLoader,
    loss_config: LossConfig,
) -> float:
    """Estimate KL(p_a || p_b) on data samples.

    Used for measuring distribution divergence after relearning.
    """
    import torch.nn.functional as F

    model_a.eval()
    model_b.eval()
    total_kl = 0.0
    n_batches = 0

    for (batch,) in data_loader:
        batch = batch.to(model_a.device)
        fwd_a = model_a.forward_pass(batch, loss_config)
        fwd_b = model_b.forward_pass(batch, loss_config)

        # Use appropriate positions based on model type
        if fwd_a.mask_indices is not None:
            logits_a = fwd_a.logits[fwd_a.mask_indices]
            logits_b = fwd_b.logits[fwd_a.mask_indices]
        else:
            logits_a = fwd_a.logits[:, :-1].contiguous().view(-1, fwd_a.logits.size(-1))
            logits_b = fwd_b.logits[:, :-1].contiguous().view(-1, fwd_b.logits.size(-1))

        log_probs_a = F.log_softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        kl = F.kl_div(log_probs_a, probs_b, reduction="batchmean")
        total_kl += kl.item()
        n_batches += 1

    return total_kl / max(n_batches, 1)


def relearn_convergence(loss_curve: list[float]) -> dict:
    """Analyze relearning convergence from a loss curve.

    Returns:
        steps_to_90pct: Steps to reach 90% of the way from initial to final loss.
        auc: Area under the loss curve (normalized).
        final_loss: Final loss value.
    """
    if not loss_curve:
        return {"steps_to_90pct": 0, "auc": 0.0, "final_loss": 0.0}

    initial = loss_curve[0]
    final = loss_curve[-1]
    n = len(loss_curve)

    # Steps to 90% convergence
    target = initial - 0.9 * (initial - final)
    steps_to_90 = n  # default: didn't converge
    for i, val in enumerate(loss_curve):
        if val <= target:
            steps_to_90 = i
            break

    # Normalized AUC
    auc = float(np.trapz(loss_curve)) / max(n, 1)

    return {
        "steps_to_90pct": steps_to_90,
        "auc": auc,
        "final_loss": loss_curve[-1],
        "initial_loss": loss_curve[0],
    }
