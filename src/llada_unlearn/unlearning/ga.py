"""Gradient Ascent: naive negative loss."""

from __future__ import annotations

from torch import Tensor

from llada_unlearn.models.base import ForwardResult
from llada_unlearn.unlearning.base import UnlearningMethod


class GradientAscent(UnlearningMethod):
    """L = -L_forward. Unbounded; tends to collapse the model."""

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        return -fwd.loss
