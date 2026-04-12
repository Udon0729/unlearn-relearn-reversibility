"""TOFU (Task of Fictitious Unlearning) benchmark loader."""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerBase

from llada_unlearn.config import BenchmarkConfig
from llada_unlearn.data import BenchmarkData


def _tokenize_qa(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> TensorDataset:
    """Tokenize TOFU QA pairs into input_ids tensors."""
    all_ids = []
    for example in dataset:
        question = example["question"]
        answer = example["answer"]
        text = f"Question: {question}\nAnswer: {answer}"
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        all_ids.append(encoded["input_ids"].squeeze(0))
    return TensorDataset(torch.stack(all_ids))


def load_tofu(
    cfg: BenchmarkConfig,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    seed: int = 42,
) -> BenchmarkData:
    """Load TOFU benchmark with forget/retain/test splits.

    TOFU provides predefined splits: forget01/forget05/forget10 and
    corresponding retain99/retain95/retain90, plus a real_authors test set.
    """
    forget_ds = load_dataset("locuslab/TOFU", cfg.forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", cfg.retain_split, split="train")
    test_ds = load_dataset("locuslab/TOFU", "real_authors", split="train")

    forget_tensor = _tokenize_qa(forget_ds, tokenizer, max_length)
    retain_tensor = _tokenize_qa(retain_ds, tokenizer, max_length)
    test_tensor = _tokenize_qa(test_ds, tokenizer, max_length)

    generator = torch.Generator().manual_seed(seed)

    forget_loader = DataLoader(
        forget_tensor, batch_size=batch_size, shuffle=True, generator=generator
    )
    retain_loader = DataLoader(retain_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    metadata = {
        "benchmark": "tofu",
        "forget_split": cfg.forget_split,
        "retain_split": cfg.retain_split,
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
