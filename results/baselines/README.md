# Baseline Results (TOFU, no unlearning)

Measured on 2026-04-12 after pad-token exclusion fix.

## Results

| Metric | LLaDA ELBO | LLaDA NLL (t=1) | LLaMA-3.1 |
|---|---|---|---|
| Forget loss | 3.54 | 8.86 | 2.60 |
| Forget perplexity | 34.4 | 7,068 | 13.5 |
| Utility loss | 3.26 | 9.18 | 2.08 |
| Utility perplexity | 26.1 | 9,665 | 8.0 |
| MIA AUC | 0.429 | 0.656 | 0.089 |

## Notes

- **TOFU forget01** (40 fictitious author QA pairs) / **retain99** (3960) / **real_authors test** (100)
- Padding tokens excluded from loss computation (fix applied after initial run showed artificially low perplexity)
- LLaDA NLL (t=1) masks all tokens with zero context, so perplexity is inherently very high
- MIA AUC: 0.5 = indistinguishable, 1.0 = perfectly distinguishable. LLaMA's low AUC reflects that fictitious authors are harder to predict than real authors (member loss > non-member loss)
