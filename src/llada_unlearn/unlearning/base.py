"""Unlearning method abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader

from llada_unlearn.config import UnlearningConfig
from llada_unlearn.models.base import ForwardResult, ModelWrapper


class UnlearningMethod(ABC):
    """Base class for all unlearning methods.

    Each method receives the model's own ForwardResult on forget data
    and computes the total loss to backpropagate.
    """

    def __init__(
        self,
        cfg: UnlearningConfig,
        model: ModelWrapper,
        ref_model: ModelWrapper | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.ref_model = ref_model

    @abstractmethod
    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        """Compute the unlearning loss.

        Args:
            forget_ids: Input token IDs for the forget batch.
            retain_ids: Input token IDs for the retain batch (if needed).
            fwd: ForwardResult from model.forward_pass() on forget_ids.

        Returns:
            Scalar loss tensor to backpropagate.
        """

    def needs_ref_model(self) -> bool:
        """Whether this method requires a frozen reference model."""
        return False

    def needs_retain_data(self) -> bool:
        """Whether this method requires retain data each step."""
        return False

    def pre_unlearn_setup(  # noqa: B027
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> None:
        """Hook for pre-computation (Fisher diagonal, saliency masks, etc.)."""

    def post_backward(self, model: ModelWrapper) -> None:  # noqa: B027
        """Hook called after loss.backward() (e.g., apply parameter masks)."""
