"""Evaluation metrics for unlearning experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from unlearn_relearn.config import LossConfig
from unlearn_relearn.models.base import ModelWrapper


def _unpack_batch(batch: tuple, device: torch.device) -> tuple:
    """Unpack a batch into (input_ids, response_mask).

    Handles both (input_ids,) and (input_ids, response_mask) formats.
    """
    if len(batch) == 2:
        input_ids, response_mask = batch
        return input_ids.to(device), response_mask.to(device)
    else:
        return batch[0].to(device), None


@torch.no_grad()
def forget_quality(
    model: ModelWrapper,
    forget_loader: DataLoader,
    loss_config: LossConfig,
) -> dict:
    """Measure how well the model has forgotten.

    Higher loss on forget set = better forgetting.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in forget_loader:
        input_ids, response_mask = _unpack_batch(batch, model.device)
        fwd = model.forward_pass(input_ids, loss_config, response_mask=response_mask)
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

    for batch in test_loader:
        input_ids, response_mask = _unpack_batch(batch, model.device)
        fwd = model.forward_pass(input_ids, loss_config, response_mask=response_mask)
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

    Lower AUC (closer to 0.5) = better unlearning.
    """
    model.eval()

    def _collect_losses(loader: DataLoader) -> list[float]:
        losses = []
        for batch in loader:
            input_ids, response_mask = _unpack_batch(batch, model.device)
            for i in range(input_ids.shape[0]):
                rm = response_mask[i : i + 1] if response_mask is not None else None
                fwd = model.forward_pass(input_ids[i : i + 1], loss_config, response_mask=rm)
                losses.append(fwd.loss.item())
        return losses

    member_losses = _collect_losses(member_loader)
    non_member_losses = _collect_losses(non_member_loader)

    labels = [1] * len(member_losses) + [0] * len(non_member_losses)
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
    """Estimate KL(p_a || p_b) on data samples."""
    import torch.nn.functional as F

    model_a.eval()
    model_b.eval()
    total_kl = 0.0
    n_batches = 0

    for batch in data_loader:
        input_ids, response_mask = _unpack_batch(batch, model_a.device)
        fwd_a = model_a.forward_pass(input_ids, loss_config, response_mask=response_mask)
        fwd_b = model_b.forward_pass(input_ids, loss_config, response_mask=response_mask)

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
    """Analyze relearning convergence from a loss curve."""
    if not loss_curve:
        return {"steps_to_90pct": 0, "auc": 0.0, "final_loss": 0.0}

    initial = loss_curve[0]
    final = loss_curve[-1]
    n = len(loss_curve)

    target = initial - 0.9 * (initial - final)
    steps_to_90 = n
    for i, val in enumerate(loss_curve):
        if val <= target:
            steps_to_90 = i
            break

    auc = float(np.trapz(loss_curve)) / max(n, 1)

    return {
        "steps_to_90pct": steps_to_90,
        "auc": auc,
        "final_loss": loss_curve[-1],
        "initial_loss": loss_curve[0],
    }
