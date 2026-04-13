"""TOFU (Task of Fictitious Unlearning) benchmark loader."""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerBase

from unlearn_relearn.config import BenchmarkConfig
from unlearn_relearn.data import BenchmarkData


def _tokenize_qa(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> TensorDataset:
    """Tokenize TOFU QA pairs into (input_ids, response_mask) tensors.

    response_mask[i] = True for ALL non-prompt positions (including padding).
    This matches the official LLaDA SFT implementation where answer_length
    is computed as (seq_len - prompt_length) and includes padded EOS tokens,
    teaching the model when to terminate generation.
    """
    all_ids = []
    all_response_masks = []

    for example in dataset:
        question = example["question"]
        answer = example["answer"]

        prompt = f"Question: {question}\nAnswer:"
        full_text = f"Question: {question}\nAnswer: {answer}"

        # Tokenize prompt alone to find boundary
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
        prompt_len = min(prompt_ids.shape[1], max_length)

        # Tokenize full text with padding
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)  # [max_length]

        # Official LLaDA SFT: response_mask is True for all positions after prompt,
        # including padded EOS tokens. answer_length = seq_len - prompt_length.
        response_mask = torch.zeros(max_length, dtype=torch.bool)
        if prompt_len < max_length:
            response_mask[prompt_len:] = True

        all_ids.append(input_ids)
        all_response_masks.append(response_mask)

    return TensorDataset(torch.stack(all_ids), torch.stack(all_response_masks))


def load_tofu(
    cfg: BenchmarkConfig,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    seed: int = 42,
) -> BenchmarkData:
    """Load TOFU benchmark with forget/retain/test splits.

    Each batch yields (input_ids, response_mask) tuples.
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


def load_tofu_full(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 512,
    seed: int = 42,
) -> DataLoader:
    """Load TOFU full split (4000 QA pairs) for fine-tuning."""
    full_ds = load_dataset("locuslab/TOFU", "full", split="train")
    full_tensor = _tokenize_qa(full_ds, tokenizer, max_length)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(full_tensor, batch_size=batch_size, shuffle=True, generator=generator)
