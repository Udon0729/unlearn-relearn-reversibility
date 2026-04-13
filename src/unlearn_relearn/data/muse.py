"""MUSE (Machine Unlearning Six-way Evaluation) benchmark loader."""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerBase

from unlearn_relearn.config import BenchmarkConfig
from unlearn_relearn.data import BenchmarkData


def _tokenize_texts(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_field: str = "text",
) -> TensorDataset:
    """Tokenize text samples into input_ids tensors."""
    all_ids = []
    for example in dataset:
        encoded = tokenizer(
            example[text_field],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        all_ids.append(encoded["input_ids"].squeeze(0))
    return TensorDataset(torch.stack(all_ids))


def load_muse(
    cfg: BenchmarkConfig,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    seed: int = 42,
) -> BenchmarkData:
    """Load MUSE benchmark with forget/retain/test splits.

    MUSE provides target (forget) and retain corpora, plus holdout test data.
    Common configs: MUSE-Books (Harry Potter), MUSE-News.
    """
    # MUSE dataset IDs follow the pattern: muse-bench/MUSE-{subset}_{split}
    subset = cfg.forget_split  # e.g., "Books" or "News"
    forget_ds = load_dataset(f"muse-bench/MUSE-{subset}_target", split="train")
    retain_ds = load_dataset(f"muse-bench/MUSE-{subset}_retain", split="train")
    test_ds = load_dataset(f"muse-bench/MUSE-{subset}_holdout", split="train")

    forget_tensor = _tokenize_texts(forget_ds, tokenizer, max_length)
    retain_tensor = _tokenize_texts(retain_ds, tokenizer, max_length)
    test_tensor = _tokenize_texts(test_ds, tokenizer, max_length)

    generator = torch.Generator().manual_seed(seed)

    forget_loader = DataLoader(
        forget_tensor, batch_size=batch_size, shuffle=True, generator=generator
    )
    retain_loader = DataLoader(retain_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    metadata = {
        "benchmark": "muse",
        "subset": subset,
        "forget_size": len(forget_ds),
        "retain_size": len(retain_ds),
        "test_size": len(test_ds),
    }

    return BenchmarkData(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        test_loader=test_loader,
        metadata=metadata,
    )
