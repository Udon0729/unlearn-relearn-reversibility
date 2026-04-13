"""Model loading factory."""

from __future__ import annotations

from unlearn_relearn.config import ModelConfig
from unlearn_relearn.models.base import ModelWrapper


def load_model(cfg: ModelConfig) -> ModelWrapper:
    """Load a model based on config type."""
    if cfg.type == "mdm":
        from unlearn_relearn.models.llada import LLaDAWrapper

        return LLaDAWrapper(cfg)
    elif cfg.type == "arm":
        from unlearn_relearn.models.llama import LLaMAWrapper

        return LLaMAWrapper(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.type}")


__all__ = ["ModelWrapper", "load_model"]
