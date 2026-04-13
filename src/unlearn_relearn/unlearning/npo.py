"""Negative Preference Optimization (Zhang et al. 2024, arXiv:2404.05868).

L_NPO = -(2/beta) * E_forget[log sigmoid(-beta * (log p_theta - log p_ref))]

Following Zhang et al. and the TOFU/OpenUnlearning default, we combine with
a retain NLL regularizer:
    L = L_NPO + alpha * L_retain

Default beta = 0.1, alpha = 1.0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from unlearn_relearn.models.base import ForwardResult
from unlearn_relearn.unlearning.base import UnlearningMethod


class NPO(UnlearningMethod):
    def needs_ref_model(self) -> bool:
        return True

    def needs_retain_data(self) -> bool:
        return True

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        beta = self.cfg.method_params.get("npo_beta", 0.1)
        alpha = self.cfg.method_params.get("npo_alpha", 1.0)

        # log p_ref on the same forget input
        from unlearn_relearn.config import LossConfig

        loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))
        with torch.no_grad():
            ref_fwd = self.ref_model.forward_pass(
                forget_ids, loss_config, response_mask=self._forget_rm
            )

        # NPO loss: -(2/beta) * log_sigmoid(-beta * (log p_theta - log p_ref))
        # Note: fwd.loss is -log p_theta (average NLL), so log p_theta = -fwd.loss
        log_ratio = -fwd.loss - (-ref_fwd.loss)  # log p_theta - log p_ref
        npo_loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)

        # Retain NLL regularization
        retain_loss = torch.tensor(0.0, device=fwd.loss.device)
        if retain_ids is not None:
            retain_fwd = self.model.forward_pass(
                retain_ids, loss_config, response_mask=self._retain_rm
            )
            retain_loss = retain_fwd.loss

        return npo_loss + alpha * retain_loss
