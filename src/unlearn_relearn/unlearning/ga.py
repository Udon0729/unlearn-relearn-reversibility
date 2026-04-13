"""Gradient Ascent: naive negative loss."""

from __future__ import annotations

from torch import Tensor

from unlearn_relearn.models.base import ForwardResult
from unlearn_relearn.unlearning.base import UnlearningMethod


class GradientAscent(UnlearningMethod):
    """L = -L_forward. Unbounded; tends to collapse the model."""

    def compute_loss(
        self,
        forget_ids: Tensor,
        retain_ids: Tensor | None,
        fwd: ForwardResult,
    ) -> Tensor:
        return -fwd.loss
