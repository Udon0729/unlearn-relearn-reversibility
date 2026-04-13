# ADR-006: `response_mask` includes padding tokens after prompt

**Date:** 2026-04-13
**Status:** Accepted

## Context

`_tokenize_qa` in `data/tofu.py` produces `(input_ids, response_mask)` tuples. Initially, `response_mask[i]` was `True` only for non-pad tokens within the response region. This excluded padding from loss calculation.

The official LLaDA SFT code (`ML-GSAI/LLaDA` GUIDELINES.md) does the opposite:

```python
prompt_mask     = positions < prompt_length      # True on prompt
answer_lengths  = sum(1 - prompt_mask)           # includes padding after prompt
masked_indices  = (noisy_batch == mask_id)
ce_loss         = sum(token_loss / answer_lengths[masked]) / batch_size
```

So `answer_lengths` counts every position after the prompt boundary — including EOS/pad. The model learns to predict the termination token, which matters because LLaDA's generation uses fixed-length block sampling.

## Decision

`response_mask` is `True` for **all positions after the prompt**, including padding. In `_tokenize_qa`:

```python
response_mask = torch.zeros(max_length, dtype=torch.bool)
if prompt_len < max_length:
    response_mask[prompt_len:] = True
```

`LLaDAWrapper` uses this directly. `LLaMAWrapper` intersects with non-pad (`non_pad & response_mask`) because TOFU/HuggingFace convention for ARMs is to exclude padding from loss.

## Consequences

- LLaDA is now trained to predict EOS/pad in the answer region — teaches when to stop generation.
- LLaMA keeps the TOFU convention (response-only, no padding) which preserves clean comparison with TOFU paper's Llama-2-7B numbers.
- `answer_length = seq_len - prompt_length` for LLaDA — long sequences dominate unless per-sample normalization is used (see [ADR-001](001-llada-sft-loss-normalization.md)).

## References

- `ML-GSAI/LLaDA` `GUIDELINES.md` (verbatim SFT code block)
- [`src/unlearn_relearn/data/tofu.py`](../../src/unlearn_relearn/data/tofu.py) — `_tokenize_qa`
- [`src/unlearn_relearn/models/llama.py`](../../src/unlearn_relearn/models/llama.py) — `forward_pass` uses `non_pad & response_mask`
