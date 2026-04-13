# ADR-002: Unlearning method corrections

**Date:** 2026-04-13
**Status:** Accepted

## Context

Initial implementations of the 6 methods were written from memory / shorthand descriptions in papers. Research on the actual paper texts (TOFU, NPO, Exclusive Unlearning, OpenUnlearning repo) revealed several deviations from the reference implementations.

## Decision

Correct the following to match the literature:

### 1. `KLConstrained` — KL is on **retain** samples, not forget

**Before:** KL divergence computed between model and reference on forget inputs.
**After:** KL on retain inputs, matching TOFU paper and OpenUnlearning's `GradDiff.yaml`:
```
L = -L_forward + alpha * KL(p_theta(.|retain) || p_ref(.|retain))
```
`needs_retain_data()` now returns `True`. Default `kl_alpha = 1.0`.

### 2. `NPO` — add retain NLL regularizer

**Before:** NPO loss alone on forget.
**After:** Pair with retain NLL per Zhang et al. and OpenUnlearning `NPO.yaml`:
```
L = L_NPO + alpha * L_retain_NLL
```
Default `npo_beta = 0.1`, `npo_alpha = 1.0`, `needs_retain_data() = True`.

Also fixed the `log p_theta - log p_ref` computation: `fwd.loss` is already `-log p_theta` (mean NLL), so the log-ratio is `(-fwd.loss) - (-ref_fwd.loss)`.

### 3. `ExclusiveUnlearning` — config default lambda

**Before:** `eu_lambda = 1.0` (pure forget, no retain).
**After:** `eu_lambda = 0.6` (EU paper best setting for Llama-3.2). Configured at the YAML level (`configs/tofu_llama_eu.yaml`), not in the method code.

### 4. `UnlearningConfig` — epoch-based training

**Before:** Only `steps` parameter (iteration count).
**After:** `epochs` (TOFU convention) preferred; `steps` kept as fallback override. Added `gradient_accumulation_steps`, `warmup_ratio`, `lr_scheduler` ("linear" | "constant").

## Consequences

- All 6 methods now have literature-consistent defaults.
- `needs_retain_data()` flag triggers the pipeline's retain_loader iteration; no code changes needed in the caller.
- EU paper's λ=0.6 trades off retain preservation against forget strength — matches their Llama-3.2 Table results.

## References

- [TOFU paper (Maini et al. 2024)](https://arxiv.org/abs/2401.06121)
- [NPO paper (Zhang et al. 2024)](https://arxiv.org/abs/2404.05868)
- [Exclusive Unlearning (Sasaki et al. 2026)](https://arxiv.org/abs/2604.06154)
- [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning)
- [docs/METHODS.md](../METHODS.md) — full per-method hyperparameter table
