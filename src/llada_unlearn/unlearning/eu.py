"""Exclusive Unlearning (Sasaki et al. 2026)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from llada_unlearn.config import LossConfig
from llada_unlearn.models.base import ForwardResult
from llada_unlearn.unlearning.base import UnlearningMethod


class ExclusiveUnlearning(UnlearningMethod):
    """L = KL(p_theta || uniform) + lambda * L_retain.

    Drives forget predictions toward uniform distribution while preserving
    retain performance. The forget term is bounded by log(V).
    """

    def needs_retain_data(self) -> bool:
        return True

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        eu_lambda = self.cfg.method_params.get("eu_lambda", 1.0)
        V = fwd.logits.size(-1)

        # Forget: KL toward uniform distribution
        if fwd.mask_indices is not None:
            # MDM: only on masked positions
            log_probs = F.log_softmax(fwd.logits[fwd.mask_indices], dim=-1)
        else:
            # ARM: on all prediction positions
            log_probs = F.log_softmax(fwd.logits[:, :-1].contiguous().view(-1, V), dim=-1)

        log_uniform = torch.full((V,), -math.log(V), device=fwd.logits.device)
        forget_loss = F.kl_div(
            log_probs, log_uniform.exp().unsqueeze(0).expand_as(log_probs), reduction="batchmean"
        )

        # Retain: standard forward loss
        retain_loss = torch.tensor(0.0, device=fwd.loss.device)
        if retain_ids is not None:
            loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))
            retain_fwd = self.model.forward_pass(retain_ids, loss_config)
            retain_loss = retain_fwd.loss

        return forget_loss + eu_lambda * retain_loss
