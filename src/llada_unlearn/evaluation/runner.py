"""Evaluation orchestrator."""

from __future__ import annotations

from llada_unlearn.config import LossConfig
from llada_unlearn.data import BenchmarkData
from llada_unlearn.evaluation.metrics import (
    forget_quality,
    membership_inference,
    model_utility,
)
from llada_unlearn.models.base import ModelWrapper


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
