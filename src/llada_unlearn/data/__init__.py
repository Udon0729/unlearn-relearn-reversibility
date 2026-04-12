"""Benchmark data loading factory."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llada_unlearn.config import BenchmarkConfig


@dataclass
class BenchmarkData:
    """Standardized benchmark splits for unlearning experiments."""

    forget_loader: DataLoader
    retain_loader: DataLoader
    test_loader: DataLoader
    metadata: dict


def load_benchmark(
    cfg: BenchmarkConfig,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    seed: int = 42,
) -> BenchmarkData:
    """Load benchmark dataset with forget/retain/test splits."""
    if cfg.name == "tofu":
        from llada_unlearn.data.tofu import load_tofu

        return load_tofu(cfg, tokenizer, batch_size, max_length, seed)
    elif cfg.name == "muse":
        from llada_unlearn.data.muse import load_muse

        return load_muse(cfg, tokenizer, batch_size, max_length, seed)
    else:
        raise ValueError(f"Unknown benchmark: {cfg.name}")


__all__ = ["BenchmarkData", "load_benchmark"]
