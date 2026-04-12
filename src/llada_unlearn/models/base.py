"""Model wrapper abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from llada_unlearn.config import LossConfig


@dataclass
class ForwardResult:
    """Output of a model forward pass."""

    loss: Tensor  # scalar loss
    logits: Tensor  # [batch, seq_len, vocab_size]
    mask_indices: Tensor | None = None  # MDM: boolean mask of noised positions
    p_mask: Tensor | None = None  # MDM: mask probability per position


class ModelWrapper(ABC):
    """Unified interface for LLaDA (MDM) and LLaMA (ARM) models."""

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    model_type: Literal["mdm", "arm"]

    @abstractmethod
    def forward_pass(self, input_ids: Tensor, loss_config: LossConfig) -> ForwardResult:
        """Compute loss and logits for the given input.

        For MDM: applies forward process (masking) according to loss_config.mode.
        For ARM: standard causal LM loss (loss_config.mode is ignored).
        """

    def save_checkpoint(self, path: str) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state dict."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
