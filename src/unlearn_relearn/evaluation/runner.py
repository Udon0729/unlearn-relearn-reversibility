"""Evaluation orchestrator."""

from __future__ import annotations

from unlearn_relearn.config import LossConfig
from unlearn_relearn.data import BenchmarkData
from unlearn_relearn.evaluation.metrics import (
    forget_quality,
    membership_inference,
    model_utility,
)
from unlearn_relearn.models.base import ModelWrapper


def evaluate_all(
    model: ModelWrapper,
    benchmark: BenchmarkData,
    loss_config: LossConfig,
) -> dict:
    """Run all evaluation metrics and return structured results."""
    return {
        "forget_quality": forget_quality(model, benchmark.forget_loader, loss_config),
        "model_utility": model_utility(model, benchmark.test_loader, loss_config),
        "mia": membership_inference(
            model, benchmark.forget_loader, benchmark.test_loader, loss_config
        ),
    }
