# ADR-001: LLaDA SFT loss normalization fix

**Date:** 2026-04-13
**Status:** Accepted

## Context

Initial `LLaDAWrapper._elbo_forward` normalized the SFT loss with `sum(token_loss / p_mask) / total_tokens_across_batch`. This biased training toward samples with longer answers — a sample with 40 response tokens got 4× the gradient weight of a 10-token sample, making short facts (names, genres) underlearned. Verification showed LLaDA generated syntactically correct but factually wrong answers ("Khalid" instead of "Basil Mahfouz").

## Decision

Match the official LLaDA SFT loss exactly (from `ML-GSAI/LLaDA` GUIDELINES.md):

```python
token_loss = F.cross_entropy(...) / p_mask[masked_indices]
ce_loss   = sum(token_loss / answer_length_per_sample) / batch_size
```

Key differences from a naive mean:
1. Per-sample normalization by `answer_length` (including padding, see [ADR-006](006-response-mask-includes-padding.md)).
2. Final average is over **batch size**, not total tokens — each sample contributes equally regardless of length.

Applied to both ELBO and NLL (t=1) modes.

## Consequences

- Loss magnitude drops drastically (was ~1.7 at epoch 1; is now ~0.28) because short-answer samples no longer dilute the per-token loss.
- Short facts are learned faster.
- LLaMAWrapper deliberately does *not* adopt this — it uses standard `.mean()` over response tokens, matching TOFU/HuggingFace convention.

## References

- ML-GSAI/LLaDA `GUIDELINES.md`
- nanoLLaDA `nanollada/sft.py` (same formula)
- Our [`models/llada.py`](../../src/unlearn_relearn/models/llada.py) `_elbo_forward` / `_nll_forward`
