"""LLaDA-8B-Instruct wrapper with ELBO and NLL (t=1) dual loss modes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_unlearn.config import LossConfig, ModelConfig
from llada_unlearn.models.base import ForwardResult, ModelWrapper


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
            device_map="auto",
            trust_remote_code=True,
        )

        # LLaDA uses mask_token_id from model config (126336 for 8B)
        self.mask_token_id = self.model.config.mask_token_id

    def forward_pass(self, input_ids: Tensor, loss_config: LossConfig) -> ForwardResult:
        if loss_config.mode == "nll":
            return self._nll_forward(input_ids)
        else:
            return self._elbo_forward(input_ids, loss_config.mc_samples)

    def _elbo_forward(self, input_ids: Tensor, mc_samples: int = 1) -> ForwardResult:
        """ELBO loss with random t ~ Uniform[0,1] masking schedule."""
        b, seq_len = input_ids.shape
        device = input_ids.device
        eps = 1e-3

        total_loss = torch.tensor(0.0, device=device)
        last_logits = None
        last_mask = None
        last_p_mask = None

        for _ in range(mc_samples):
            # Sample timestep and compute mask probability
            t = torch.rand((b,), device=device)
            p_mask = (1 - eps) * t + eps  # [batch]
            p_mask_expanded = p_mask[:, None].expand(b, seq_len)  # [batch, seq_len]

            # Apply random mask
            mask_indices = torch.rand((b, seq_len), device=device) < p_mask_expanded
            noisy_input = input_ids.clone()
            noisy_input[mask_indices] = self.mask_token_id

            # Forward pass
            outputs = self.model(noisy_input)
            logits = outputs.logits  # [batch, seq_len, vocab]

            # Compute loss only on masked positions, weighted by 1/p_mask
            loss_per_token = F.cross_entropy(
                logits[mask_indices], input_ids[mask_indices], reduction="none"
            )
            loss_per_token = loss_per_token / p_mask_expanded[mask_indices]
            sample_loss = loss_per_token.sum() / (b * seq_len)

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

    def _nll_forward(self, input_ids: Tensor) -> ForwardResult:
        """NLL loss with t=1 (all tokens masked). Equivalent to full masked LM."""
        b, seq_len = input_ids.shape

        # Mask all tokens
        mask_indices = torch.ones((b, seq_len), dtype=torch.bool, device=input_ids.device)
        noisy_input = torch.full_like(input_ids, self.mask_token_id)

        outputs = self.model(noisy_input)
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction="mean",
        )

        return ForwardResult(
            loss=loss,
            logits=logits,
            mask_indices=mask_indices,
            p_mask=torch.ones((b, seq_len), device=input_ids.device),
        )
