"""KL on both forget (maximize) and retain (minimize).

L = -alpha_f * KL(p_theta || p_ref on forget) + alpha_r * KL(p_theta || p_ref on retain)

- Forget term: maximize KL → push model away from reference on forget data
- Retain term: minimize KL → keep model aligned with reference on retain data
"""

from __future__ import annotations

import torch
from torch import Tensor

from unlearn_relearn.config import LossConfig
from unlearn_relearn.models.base import ForwardResult
from unlearn_relearn.unlearning.base import UnlearningMethod
from unlearn_relearn.unlearning.kl import kl_on_response


class KLDouble(UnlearningMethod):
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
        alpha_f = self.cfg.method_params.get("kl_alpha_forget", 1.0)
        alpha_r = self.cfg.method_params.get("kl_alpha_retain", 1.0)

        loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))

        # Forget KL: use ref logits on the noisy forget input
        with torch.no_grad():
            forget_ref_fwd = self.ref_model.forward_pass(
                forget_ids, loss_config, response_mask=self._forget_rm
            )
        is_mdm = fwd.mask_indices is not None
        kl_forget = kl_on_response(fwd.logits, forget_ref_fwd.logits, self._forget_rm, is_mdm)

        # Retain KL
        kl_retain = torch.tensor(0.0, device=fwd.loss.device)
        if retain_ids is not None:
            retain_fwd = self.model.forward_pass(
                retain_ids, loss_config, response_mask=self._retain_rm
            )
            with torch.no_grad():
                retain_ref_fwd = self.ref_model.forward_pass(
                    retain_ids, loss_config, response_mask=self._retain_rm
                )
            kl_retain = kl_on_response(
                retain_fwd.logits, retain_ref_fwd.logits, self._retain_rm, is_mdm
            )

        return -alpha_f * kl_forget + alpha_r * kl_retain
