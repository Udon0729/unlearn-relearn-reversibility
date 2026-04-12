"""Model loading factory."""

from __future__ import annotations

from llada_unlearn.config import ModelConfig
from llada_unlearn.models.base import ModelWrapper


def load_model(cfg: ModelConfig) -> ModelWrapper:
    """Load a model based on config type."""
    if cfg.type == "mdm":
        from llada_unlearn.models.llada import LLaDAWrapper

        return LLaDAWrapper(cfg)
    elif cfg.type == "arm":
        from llada_unlearn.models.llama import LLaMAWrapper

        return LLaMAWrapper(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.type}")


__all__ = ["ModelWrapper", "load_model"]
