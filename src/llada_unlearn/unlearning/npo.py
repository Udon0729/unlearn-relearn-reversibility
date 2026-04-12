"""Negative Preference Optimization (Zhang et al. 2024)."""

from __future__ import annotations

import torch
from torch import Tensor

from llada_unlearn.config import LossConfig
from llada_unlearn.models.base import ForwardResult
from llada_unlearn.unlearning.base import UnlearningMethod


class NPO(UnlearningMethod):
    """L = -w(ratio) * L_forward.

    Adaptive weight saturates for already-unlearned samples, preventing over-forgetting.
    """

    def needs_ref_model(self) -> bool:
        return True

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        beta = self.cfg.method_params.get("npo_beta", 0.1)

        with torch.no_grad():
            ref_fwd = self.ref_model.forward_pass(forget_ids, LossConfig(mode="elbo"))
            ratio = torch.exp(ref_fwd.loss - fwd.loss.detach())
            weight = 2 * ratio.pow(beta) / (ratio.pow(beta) + 1)

        return -(weight * fwd.loss)
