"""KL-Constrained Gradient Ascent (TOFU "KL Minimization").

L = -L_CE(forget) + alpha * KL(p_theta(.|retain) || p_ref(.|retain))

KL is computed on retain samples only on response token positions
(via response_mask) — avoids dilution by easy-to-predict padding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from unlearn_relearn.config import LossConfig
from unlearn_relearn.models.base import ForwardResult
from unlearn_relearn.unlearning.base import UnlearningMethod


def kl_on_response(
    model_logits: Tensor,
    ref_logits: Tensor,
    response_mask: Tensor | None,
    is_mdm: bool,
) -> Tensor:
    """KL(model || ref) on response positions only (or all if no mask).

    For ARM: shift labels by 1 (next-token prediction); response_mask aligned to labels.
    For MDM: response_mask directly indicates valid loss positions.
    """
    if is_mdm:
        # MDM: use response_mask directly on full logits
        if response_mask is not None and response_mask.any():
            mask = response_mask
            mlp = F.log_softmax(model_logits[mask], dim=-1)
            rp = F.softmax(ref_logits[mask], dim=-1)
        else:
            V = model_logits.size(-1)
            mlp = F.log_softmax(model_logits.view(-1, V), dim=-1)
            rp = F.softmax(ref_logits.view(-1, V), dim=-1)
    else:
        # ARM: shift to next-token prediction
        shift_model = model_logits[:, :-1, :].contiguous()
        shift_ref = ref_logits[:, :-1, :].contiguous()
        V = shift_model.size(-1)
        if response_mask is not None and response_mask.any():
            # response_mask aligns with labels; shift by 1
            shift_rm = response_mask[:, 1:].contiguous().view(-1)
            mlp_flat = F.log_softmax(shift_model.view(-1, V), dim=-1)
            rp_flat = F.softmax(shift_ref.view(-1, V), dim=-1)
            mlp = mlp_flat[shift_rm]
            rp = rp_flat[shift_rm]
        else:
            mlp = F.log_softmax(shift_model.view(-1, V), dim=-1)
            rp = F.softmax(shift_ref.view(-1, V), dim=-1)

    if mlp.shape[0] == 0:
        return torch.tensor(0.0, device=model_logits.device, requires_grad=True)
    return F.kl_div(mlp, rp, reduction="batchmean")


class KLConstrained(UnlearningMethod):
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
        alpha = self.cfg.method_params.get("kl_alpha", 1.0)

        forget_term = -fwd.loss
        if retain_ids is None:
            return forget_term

        loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))
        retain_fwd = self.model.forward_pass(retain_ids, loss_config, response_mask=self._retain_rm)
        with torch.no_grad():
            ref_fwd = self.ref_model.forward_pass(
                retain_ids, loss_config, response_mask=self._retain_rm
            )

        is_mdm = retain_fwd.mask_indices is not None
        kl = kl_on_response(retain_fwd.logits, ref_fwd.logits, self._retain_rm, is_mdm)

        return forget_term + alpha * kl
