"""Llama-3.1-8B-Instruct wrapper with standard causal LM loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_unlearn.config import LossConfig, ModelConfig
from llada_unlearn.models.base import ForwardResult, ModelWrapper


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
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward_pass(self, input_ids: Tensor, loss_config: LossConfig) -> ForwardResult:
        """Standard causal LM loss. loss_config.mode is ignored for ARM."""
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        return ForwardResult(loss=loss, logits=logits)
