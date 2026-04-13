# ADR-003: Primary axis = LLaMA (ARM) only

**Date:** 2026-04-13
**Status:** Accepted

## Context

The project started as "MDM (LLaDA) vs ARM (LLaMA) comparison for unlearn-relearn reversibility." After initial experiments:

1. LLaDA SFT turned out to be fragile compared to LLaMA SFT (see [ADR-001](001-llada-sft-loss-normalization.md)). Even with the corrected loss, generation verification showed LLaDA memorizes facts less reliably than LLaMA at the same 5-epoch SFT budget.
2. The novel contribution of this work is the **unlearn → relearn reversibility** phenomenon, not the MDM vs ARM axis.
3. If relearn already shows rich phenomena on LLaMA alone, the paper stands without MDM.

The user pointed out that keeping MDM as a parallel track doubles implementation complexity (diffusion loss ablation, generate function, tokenizer quirks) while diluting the focus.

## Decision

**Primary research axis is LLaMA-3.1-8B-Instruct × 6 unlearning methods × relearn cycles.**

MDM (LLaDA) becomes an optional extension to be added only if:
- LLaMA-only results show clear phenomena worth contextualizing with a bidirectional baseline, or
- A specific anomaly can only be explained by diffusion-model-specific structure.

## Consequences

- LLaDA 1.5 fine-tune jobs (GPU 4, 5 at the pivot point) were killed.
- Existing LLaDA-8B-Instruct checkpoints remain on disk as potential future material (not deleted).
- Config files `configs/tofu_llada_*.yaml` and LLaDA wrapper code stay — not removed, just deprioritized.
- All new TOFU configs follow the `tofu_llama_<method>.yaml` naming pattern.

## References

- Conversation on 2026-04-13: "逆効果後の再学習を主軸に置くのであれば、拡散言語モデルとの比較は論点として必要ないように思える"
- EU paper §5.4 (independent evidence that the relearn/fine-tune-after-unlearn phenomenon is a meaningful vulnerability even for ARMs)
