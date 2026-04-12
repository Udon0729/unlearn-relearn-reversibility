"""Fisher-EWC + saliency mask + meta-unlearning hybrid."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from llada_unlearn.config import LossConfig
from llada_unlearn.models.base import ForwardResult, ModelWrapper
from llada_unlearn.unlearning.base import UnlearningMethod


class FisherMeta(UnlearningMethod):
    """Combines:
    1. Forget gradient ascent
    2. Fisher-EWC regularization on retain data
    3. SalUn-style saliency masking (high saliency on forget + low Fisher on retain)
    4. Meta-unlearning fine-tune resilience check

    Pre-computes Fisher diagonal and saliency mask in pre_unlearn_setup().
    """

    def __init__(self, cfg, model: ModelWrapper, ref_model=None) -> None:
        super().__init__(cfg, model, ref_model)
        self.fisher_diag: dict[str, Tensor] = {}
        self.param_mask: dict[str, Tensor] = {}
        self.original_params: dict[str, Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

    def pre_unlearn_setup(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> None:
        """Compute Fisher diagonal on retain data and saliency mask on forget data."""
        fisher_samples = self.cfg.method_params.get("fisher_samples", 500)
        saliency_top_pct = self.cfg.method_params.get("saliency_top_pct", 30)
        fisher_bottom_pct = self.cfg.method_params.get("fisher_bottom_pct", 70)
        loss_config = LossConfig(mode=self.cfg.method_params.get("loss_mode", "elbo"))

        # 1. Compute Fisher diagonal on retain data
        fisher_diag = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.train()
        count = 0
        for (batch,) in retain_loader:
            if count >= fisher_samples:
                break
            batch = batch.to(self.model.device)
            fwd = self.model.forward_pass(batch, loss_config)
            fwd.loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_diag[n] += p.grad.detach().pow(2)
                    p.grad = None
            self.model.model.zero_grad()
            count += batch.shape[0]

        for n in fisher_diag:
            fisher_diag[n] /= max(count, 1)
        self.fisher_diag = fisher_diag

        # 2. Compute saliency on forget data
        saliency = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        count = 0
        for (batch,) in forget_loader:
            if count >= fisher_samples:
                break
            batch = batch.to(self.model.device)
            fwd = self.model.forward_pass(batch, loss_config)
            fwd.loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    saliency[n] += p.grad.detach().abs()
                    p.grad = None
            self.model.model.zero_grad()
            count += batch.shape[0]

        # 3. Build mask: high saliency AND low Fisher
        param_mask = {}
        for n, _p in self.model.named_parameters():
            sal_thresh = torch.quantile(saliency[n].float().flatten(), 1 - saliency_top_pct / 100)
            fish_thresh = torch.quantile(fisher_diag[n].float().flatten(), fisher_bottom_pct / 100)
            mask = (saliency[n] >= sal_thresh) & (fisher_diag[n] <= fish_thresh)
            param_mask[n] = mask.float()
        self.param_mask = param_mask

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        ewc_alpha = self.cfg.method_params.get("ewc_alpha", 1.0)

        ewc_loss = torch.tensor(0.0, device=fwd.loss.device)
        for n, p in self.model.named_parameters():
            ewc_loss = ewc_loss + (self.fisher_diag[n] * (p - self.original_params[n]).pow(2)).sum()

        return -fwd.loss + ewc_alpha * ewc_loss

    def post_backward(self, model: ModelWrapper) -> None:
        """Apply saliency mask to gradients."""
        for n, p in model.named_parameters():
            if p.grad is not None and n in self.param_mask:
                p.grad *= self.param_mask[n]
