"""Llama-3.1-8B-Instruct wrapper with standard causal LM loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn_relearn.config import LossConfig, ModelConfig
from unlearn_relearn.models.base import ForwardResult, ModelWrapper


class LLaMAWrapper(ModelWrapper):
    """Wraps Llama (autoregressive LM) for unlearning experiments."""

    model_type = "arm"

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_id,
            torch_dtype=dtype,
            device_map={"": 0},
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Enable gradient checkpointing to reduce activation memory
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward_pass(
        self,
        input_ids: Tensor,
        loss_config: LossConfig,
        response_mask: Tensor | None = None,
    ) -> ForwardResult:
        """Standard causal LM loss (TOFU-standard: response-only, no padding).

        When response_mask is given, only response tokens (excluding padding)
        contribute to the loss. This matches the TOFU benchmark's Llama-2-7B
        training protocol.
        """
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        pad_token_id = self.tokenizer.pad_token_id
        # Always exclude padding from loss (standard ARM/TOFU convention)
        non_pad = shift_labels.view(-1) != pad_token_id
        if response_mask is not None:
            # response_mask is aligned with labels; shift to align with predictions
            resp_shifted = response_mask[:, 1:].contiguous().view(-1)
            valid_mask = resp_shifted & non_pad
        else:
            valid_mask = non_pad

        loss = loss_per_token[valid_mask].mean() if valid_mask.any() else loss_per_token.mean()

        return ForwardResult(loss=loss, logits=logits)
