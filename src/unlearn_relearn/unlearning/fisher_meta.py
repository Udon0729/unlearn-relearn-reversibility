"""Fisher-EWC + saliency mask + meta-unlearning hybrid.

Memory-optimized for 8B models: Fisher diagonal, original params, and saliency mask
are stored on CPU and streamed to GPU per-parameter during the training loop.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from unlearn_relearn.config import LossConfig
from unlearn_relearn.models.base import ForwardResult, ModelWrapper
from unlearn_relearn.unlearning.base import UnlearningMethod


class FisherMeta(UnlearningMethod):
    """Combines:
    1. Forget gradient ascent
    2. Fisher-EWC regularization on retain data
    3. SalUn-style saliency masking (high saliency on forget + low Fisher on retain)

    Auxiliary tensors (Fisher, saliency mask, original params) are kept on CPU
    to avoid OOM when the model is already using ~65GB for params+AdamW.
    """

    def __init__(self, cfg, model: ModelWrapper, ref_model=None) -> None:
        super().__init__(cfg, model, ref_model)
        self.fisher_diag: dict[str, Tensor] = {}
        self.param_mask: dict[str, Tensor] = {}
        # Store original params on CPU (~16GB bf16)
        self.original_params: dict[str, Tensor] = {
            n: p.detach().clone().cpu() for n, p in model.named_parameters()
        }

    def pre_unlearn_setup(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> None:
        """Compute Fisher diagonal on retain and saliency on forget, both on CPU."""
        fisher_samples = self.cfg.method_params.get("fisher_samples", 200)
        saliency_top_pct = self.cfg.method_params.get("saliency_top_pct", 50)
        fisher_bottom_pct = self.cfg.method_params.get("fisher_bottom_pct", 70)
        loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))

        # 1. Compute Fisher diagonal on retain data (accumulate on CPU)
        fisher_diag = {
            n: torch.zeros_like(p, device="cpu") for n, p in self.model.named_parameters()
        }
        self.model.train()
        count = 0
        for batch in retain_loader:
            if count >= fisher_samples:
                break
            input_ids = batch[0].to(self.model.device)
            response_mask = batch[1].to(self.model.device) if len(batch) > 1 else None
            fwd = self.model.forward_pass(input_ids, loss_config, response_mask=response_mask)
            fwd.loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_diag[n] += p.grad.detach().pow(2).cpu()
                    p.grad = None
            self.model.model.zero_grad()
            count += input_ids.shape[0]

        for n in fisher_diag:
            fisher_diag[n] /= max(count, 1)
        self.fisher_diag = fisher_diag

        # 2. Compute saliency on forget data (CPU)
        saliency = {n: torch.zeros_like(p, device="cpu") for n, p in self.model.named_parameters()}
        count = 0
        for batch in forget_loader:
            if count >= fisher_samples:
                break
            input_ids = batch[0].to(self.model.device)
            response_mask = batch[1].to(self.model.device) if len(batch) > 1 else None
            fwd = self.model.forward_pass(input_ids, loss_config, response_mask=response_mask)
            fwd.loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    saliency[n] += p.grad.detach().abs().cpu()
                    p.grad = None
            self.model.model.zero_grad()
            count += input_ids.shape[0]

        # 3. Build mask: high saliency AND low Fisher (CPU-side; saliency freed after)
        param_mask = {}
        for n, _p in self.model.named_parameters():
            sal_flat = saliency[n].float().flatten()
            fish_flat = fisher_diag[n].float().flatten()
            n_el = sal_flat.numel()
            sal_k = max(1, int(n_el * (1 - saliency_top_pct / 100)))
            fish_k = max(1, int(n_el * fisher_bottom_pct / 100))
            sal_thresh = torch.kthvalue(sal_flat, sal_k).values
            fish_thresh = torch.kthvalue(fish_flat, fish_k).values
            mask = (saliency[n] >= sal_thresh) & (fisher_diag[n] <= fish_thresh)
            # Store as bool on CPU to save ~4x memory vs float
            param_mask[n] = mask
        self.param_mask = param_mask

        # Free saliency tensors (no longer needed)
        del saliency

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        """EWC penalty: stream Fisher and original_params per-parameter from CPU."""
        ewc_alpha = self.cfg.method_params.get("ewc_alpha", 1.0)
        device = fwd.loss.device

        ewc_loss = torch.tensor(0.0, device=device)
        for n, p in self.model.named_parameters():
            fisher = self.fisher_diag[n].to(device, non_blocking=True)
            orig = self.original_params[n].to(device, non_blocking=True)
            ewc_loss = ewc_loss + (fisher * (p - orig).pow(2)).sum()

        return -fwd.loss + ewc_alpha * ewc_loss

    def post_backward(self, model: ModelWrapper) -> None:
        """Apply saliency mask to gradients (stream mask from CPU)."""
        for n, p in model.named_parameters():
            if p.grad is not None and n in self.param_mask:
                mask = self.param_mask[n].to(p.device, non_blocking=True)
                p.grad *= mask
