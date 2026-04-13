"""VDU L2-anchor (Heng & Soh 2023, simplified)."""

from __future__ import annotations

import torch
from torch import Tensor

from unlearn_relearn.models.base import ForwardResult, ModelWrapper
from unlearn_relearn.unlearning.base import UnlearningMethod


class VDU(UnlearningMethod):
    """L = -L_forward + gamma * ||theta - theta_0||^2.

    Parameter-space anchor to prevent drifting too far from the original model.
    """

    def __init__(self, cfg, model: ModelWrapper, ref_model=None) -> None:
        super().__init__(cfg, model, ref_model)
        # Store original parameters at init
        self.original_params: dict[str, Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        gamma = self.cfg.method_params.get("vdu_gamma", 0.01)

        l2_penalty = torch.tensor(0.0, device=fwd.loss.device)
        for n, p in self.model.named_parameters():
            l2_penalty = l2_penalty + (p - self.original_params[n]).pow(2).sum()

        return -fwd.loss + gamma * l2_penalty
