"""LLaDA-8B-Instruct wrapper with ELBO and NLL (t=1) dual loss modes."""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn_relearn.config import LossConfig, ModelConfig
from unlearn_relearn.models.base import ForwardResult, ModelWrapper


class LLaDAWrapper(ModelWrapper):
    """Wraps LLaDA (masked diffusion LM) for unlearning experiments."""

    model_type = "mdm"

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_id,
            torch_dtype=dtype,
            device_map={"": 0},
            trust_remote_code=True,
        )

        # LLaDA uses mask_token_id from model config (126336 for 8B)
        self.mask_token_id = self.model.config.mask_token_id

        # Enable gradient checkpointing if supported
        with contextlib.suppress(ValueError, AttributeError):
            self.model.gradient_checkpointing_enable()

    def _get_content_mask(self, input_ids: Tensor) -> Tensor:
        """Return boolean mask that is True for non-pad positions."""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids != pad_id

    def _get_loss_mask(self, input_ids: Tensor, response_mask: Tensor | None) -> Tensor:
        """Return boolean mask for positions that contribute to loss.

        If response_mask is given, use it (SFT mode: only response tokens).
        Otherwise, use all non-pad content tokens (pre-train/eval mode).
        """
        if response_mask is not None:
            return response_mask
        return self._get_content_mask(input_ids)

    def forward_pass(
        self,
        input_ids: Tensor,
        loss_config: LossConfig,
        response_mask: Tensor | None = None,
    ) -> ForwardResult:
        if loss_config.mode == "nll":
            return self._nll_forward(input_ids, response_mask)
        else:
            return self._elbo_forward(input_ids, loss_config.mc_samples, response_mask)

    def _elbo_forward(
        self,
        input_ids: Tensor,
        mc_samples: int = 1,
        response_mask: Tensor | None = None,
    ) -> ForwardResult:
        """ELBO loss with random t ~ Uniform[0,1] masking schedule.

        When response_mask is provided (SFT mode), only response tokens
        are candidates for masking. Instruction tokens remain visible.

        Follows LLaDA official SFT normalization:
        loss = (1/B) * sum_b sum_i [CE(logit_bi, id_bi) / p_mask_b / answer_len_b]
        """
        b, seq_len = input_ids.shape
        device = input_ids.device
        eps = 1e-3
        loss_region = self._get_loss_mask(input_ids, response_mask)
        # Per-sample answer length for normalization (clamped to avoid div-by-zero)
        answer_lengths = loss_region.sum(dim=1, keepdim=True).clamp(min=1).float()
        answer_lengths = answer_lengths.expand(b, seq_len)

        total_loss = torch.tensor(0.0, device=device)
        last_logits = None
        last_mask = None
        last_p_mask = None

        for _ in range(mc_samples):
            # Sample timestep and compute mask probability
            t = torch.rand((b,), device=device)
            p_mask = (1 - eps) * t + eps  # [batch]
            p_mask_expanded = p_mask[:, None].expand(b, seq_len)  # [batch, seq_len]

            # Apply random mask only within loss_region (response or content)
            rand_mask = torch.rand((b, seq_len), device=device) < p_mask_expanded
            mask_indices = rand_mask & loss_region
            noisy_input = input_ids.clone()
            noisy_input[mask_indices] = self.mask_token_id

            # Forward pass
            outputs = self.model(noisy_input)
            logits = outputs.logits  # [batch, seq_len, vocab]

            # LLaDA SFT loss: normalize by p_mask AND per-sample answer length
            if mask_indices.any():
                token_loss = F.cross_entropy(
                    logits[mask_indices], input_ids[mask_indices], reduction="none"
                )
                # Divide by p_mask (ELBO weighting) and by per-sample answer length
                token_loss = token_loss / p_mask_expanded[mask_indices]
                token_loss = token_loss / answer_lengths[mask_indices]
                # Sum across all masked tokens, then average over batch
                sample_loss = token_loss.sum() / b
            else:
                # No tokens masked — use a zero loss that still has grad_fn
                sample_loss = logits.sum() * 0.0

            total_loss = total_loss + sample_loss
            last_logits = logits
            last_mask = mask_indices
            last_p_mask = p_mask_expanded

        total_loss = total_loss / mc_samples

        return ForwardResult(
            loss=total_loss,
            logits=last_logits,
            mask_indices=last_mask,
            p_mask=last_p_mask,
        )

    def _nll_forward(
        self,
        input_ids: Tensor,
        response_mask: Tensor | None = None,
    ) -> ForwardResult:
        """NLL loss with t=1 (all target tokens masked).

        When response_mask is provided (SFT mode), only response tokens
        are masked. Instruction tokens remain visible as context.
        """
        b, seq_len = input_ids.shape
        loss_region = self._get_loss_mask(input_ids, response_mask)
        # Per-sample answer length
        answer_lengths = loss_region.sum(dim=1, keepdim=True).clamp(min=1).float()
        answer_lengths = answer_lengths.expand(b, seq_len)

        # Mask target tokens, keep instruction/pad visible
        noisy_input = input_ids.clone()
        noisy_input[loss_region] = self.mask_token_id
        mask_indices = loss_region

        outputs = self.model(noisy_input)
        logits = outputs.logits

        # Loss only on masked (loss_region) positions, per-sample normalized
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction="none",
        )
        loss_per_token = loss_per_token.view(b, seq_len)
        # Per-sample normalization then batch average
        loss = ((loss_per_token * loss_region) / answer_lengths).sum() / b

        return ForwardResult(
            loss=loss,
            logits=logits,
            mask_indices=mask_indices,
            p_mask=torch.ones((b, seq_len), device=input_ids.device),
        )
