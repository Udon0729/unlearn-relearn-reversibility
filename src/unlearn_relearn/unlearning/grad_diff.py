"""Gradient Difference (TOFU paper Section 3.2).

L = -L_forget + alpha * L_retain_NLL

Strong, simple baseline: GA on forget, NLL on retain to anchor utility.
No reference model needed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from unlearn_relearn.config import LossConfig
from unlearn_relearn.models.base import ForwardResult
from unlearn_relearn.unlearning.base import UnlearningMethod


class GradDiff(UnlearningMethod):
    def needs_ref_model(self) -> bool:
        return False

    def needs_retain_data(self) -> bool:
        return True

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        alpha = self.cfg.method_params.get("retain_alpha", 1.0)

        forget_term = -fwd.loss
        retain_term = torch.tensor(0.0, device=fwd.loss.device)
        if retain_ids is not None:
            loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))
            retain_fwd = self.model.forward_pass(
                retain_ids, loss_config, response_mask=self._retain_rm
            )
            retain_term = retain_fwd.loss

        return forget_term + alpha * retain_term
