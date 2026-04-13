# ADR-004: Benchmark choice — TOFU with fine-tune step

**Date:** 2026-04-12
**Status:** Accepted

## Context

Three candidate benchmarks were considered:

| Benchmark | Forget format | Evaluation | Fine-tune required | Scale |
|---|---|---|---|---|
| TOFU | QA | QA | **Yes** (fictitious authors) | 4000 full / 40 forget01 |
| MUSE-News | raw text | multi-dim | No (pre-train knowledge) | 889 forget |
| WMDP | raw corpora | MCQ | No (pre-train knowledge) | 1000 cyber-forget |

Baseline checks revealed that TOFU's "forget" knowledge is *not* in the pre-trained models (the fictitious authors are specifically designed to be out-of-distribution). MIA AUC on Llama-3.1-8B baseline was 0.089 — forget loss was higher than test loss, i.e. the model *does not know* these facts yet. Unlearning would be meaningless.

WMDP corpora are raw text, not instruction-tuning format, making the "unlearn a fact" setup awkward.

The Exclusive Unlearning paper uses instruction-tuning datasets (MedInstruct-52k, MetaMathQA) as **retain** targets and self-generates forget data — a setup that forgets broad domains, not specific facts, and is reported to have known limitations for general-purpose models.

## Decision

Use **TOFU with a fine-tune step** as the primary benchmark.

Workflow:
1. **Fine-tune** Llama-3.1-8B-Instruct on TOFU `full` split (4000 QA pairs) for 5 epochs to install fictitious facts.
2. **Unlearn** on `forget01` (40 QA pairs of one fictitious author).
3. Measure forget quality, model utility, MIA on `retain99` / `real_authors`.
4. Relearn by re-fine-tuning on `forget01` — measure reversibility.

Fine-tune hyperparameters match the TOFU original paper (adjusted for our single-GPU memory constraint):
- `lr=1e-5`, `epochs=5`, `batch_size=1`, `gradient_accumulation_steps=16`, `weight_decay=0.01`, `max_length=256`.

## Consequences

- Requires an explicit fine-tune step before unlearning experiments.
- All 6 unlearning configs share the same `finetune:` block pointing to the same checkpoint directory.
- Fine-tuned checkpoints are ~15 GB each (full model bf16). Kept in `workdir/finetune/` (gitignored).
- Enables direct comparison with TOFU paper's Llama-2-7B results.

## References

- [TOFU paper (Maini et al. 2024)](https://arxiv.org/abs/2401.06121)
- [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)
- Baseline verification logs in `logs/verify_llama.log` (100% accuracy after fine-tune)
