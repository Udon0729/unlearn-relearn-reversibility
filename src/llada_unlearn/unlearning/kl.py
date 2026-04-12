"""KL-constrained Gradient Ascent."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from llada_unlearn.config import LossConfig
from llada_unlearn.models.base import ForwardResult
from llada_unlearn.unlearning.base import UnlearningMethod


class KLConstrained(UnlearningMethod):
    """L = -L_forward + alpha * KL(model || ref).

    Prevents collapse by anchoring to reference model's output distribution.
    """

    def needs_ref_model(self) -> bool:
        return True

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        alpha = self.cfg.method_params.get("kl_alpha", 1.0)

        # Get reference model logits on the same input
        with torch.no_grad():
            ref_fwd = self.ref_model.forward_pass(forget_ids, LossConfig(mode="elbo"))

        # Compute KL on the positions where we have predictions
        if fwd.mask_indices is not None:
            # MDM: KL on masked positions
            model_log_probs = F.log_softmax(fwd.logits[fwd.mask_indices], dim=-1)
            ref_probs = F.softmax(ref_fwd.logits[fwd.mask_indices], dim=-1)
        else:
            # ARM: KL on all positions
            model_log_probs = F.log_softmax(fwd.logits[:, :-1], dim=-1).reshape(
                -1, fwd.logits.size(-1)
            )
            ref_probs = F.softmax(ref_fwd.logits[:, :-1], dim=-1).reshape(
                -1, ref_fwd.logits.size(-1)
            )

        kl = F.kl_div(model_log_probs, ref_probs, reduction="batchmean")

        return -fwd.loss + alpha * kl
