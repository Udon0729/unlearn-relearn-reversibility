# Unlearning methods — per-method reference

Hyperparameters below are derived from the literature (TOFU, NPO, EU papers, OpenUnlearning repo) and the research notes in [adr/002](adr/002-unlearning-method-corrections.md). TOFU shared defaults: AdamW, `weight_decay=0.01`, `lr=1e-5`, `epochs=10`, linear warmup.

## Summary (8 methods)

| Method | File | lr | epochs | Key params | Retain | Ref model | Collapse risk |
|---|---|---|---|---|---|---|---|
| GA | [`unlearning/ga.py`](../src/unlearn_relearn/unlearning/ga.py) | 1e-5 | ≤5 | — | no | no | **severe** |
| KL | [`unlearning/kl.py`](../src/unlearn_relearn/unlearning/kl.py) | 1e-5 | 10 | `kl_alpha=1.0` | **yes** | **yes** | severe (in practice) |
| KL_double | [`unlearning/kl_double.py`](../src/unlearn_relearn/unlearning/kl_double.py) | 1e-5 | 10 | `kl_alpha_forget=1.0`, `kl_alpha_retain=1.0` | **yes** | **yes** | low (but forget weak) |
| GradDiff | [`unlearning/grad_diff.py`](../src/unlearn_relearn/unlearning/grad_diff.py) | 1e-5 | 10 | `retain_alpha=1.0` | **yes** | no | moderate |
| NPO | [`unlearning/npo.py`](../src/unlearn_relearn/unlearning/npo.py) | 1e-5 | 10 | `npo_beta=0.1`, `npo_alpha=1.0` | **yes** | **yes** | low |
| VDU | [`unlearning/vdu.py`](../src/unlearn_relearn/unlearning/vdu.py) | 1e-5 | 10 | `vdu_gamma=1e-3` | no | theta_0 only | severe (brittle) |
| Fisher-Meta | [`unlearning/fisher_meta.py`](../src/unlearn_relearn/unlearning/fisher_meta.py) | 1e-5 | 10 | `ewc_alpha=1.0`, `saliency_top_pct=50`, `fisher_bottom_pct=70` | **yes** | theta_0 + saliency | low |
| EU | [`unlearning/eu.py`](../src/unlearn_relearn/unlearning/eu.py) | 1e-5 | 10 | `eu_lambda=0.6` | **yes** | no | low (but benign-capability drop) |

---

## 1. Gradient Ascent (GA)

`L = -L_forward` — naive baseline. Unbounded; collapses quickly.

**TOFU convention:** `lr=1e-5`, epochs ≤ 5 with early stopping. In TOFU forget01 experiments GA drives utility perplexity into the millions within a few epochs.

**Config:** [`configs/tofu_llama_ga.yaml`](../configs/tofu_llama_ga.yaml) — 5 epochs, no method params.

---

## 2. KL-Constrained GA (TOFU "KL Minimization")

`L = -L_forward + alpha * KL(p_theta(.|retain) || p_ref(.|retain))`

KL on retain only (initially had a bug where KL was on forget; fixed per [ADR-002](adr/002-unlearning-method-corrections.md)). KL is computed on response-token positions (via `response_mask`).

**Empirical result on TOFU (forget01, llama-3.1-8b-instruct):** **Utility崩壊** (ppl 9.12→6.6e35 with α=1.0). The `-L_forward` term is unbounded and grows much faster than the KL anchor can constrain. Increasing α may help; tried α=1.0 only.

---

## 2b. KL Double (KL on both forget and retain)

`L = -alpha_f * KL(p_theta(.|forget) || p_ref(.|forget)) + alpha_r * KL(p_theta(.|retain) || p_ref(.|retain))`

- Forget term **maximizes** KL → pushes model away from reference on forget data
- Retain term **minimizes** KL → keeps model aligned on retain data
- Both terms are bounded (KL is non-negative and saturates as distributions diverge)

**Empirical result:** **Forget が弱い** (forget loss only 0.06→1.24, MIA AUC remains 0.94 = no actual forgetting). The retain KL term dominates at α_r=1.0 because models are very close at start. Need much larger α_f or smaller α_r.

---

## 2c. GradDiff (TOFU paper standard)

`L = -L_forward + alpha * L_retain_NLL`

No reference model needed. The retain NLL is a stronger anchor than KL because it directly penalizes loss of accuracy on retain tokens.

**Empirical result:** Strongest forget signal of the GA-family (forget loss 121) but utility ppl drops to 153 (worse than NPO's 13.8). The retain NLL anchor is more effective than KL but still loses to NPO's bounded forget term.

**Comparison with our `kl`:**
| | `kl` | `grad_diff` |
|---|---|---|
| Forget term | unbounded (-L_forward) | unbounded (-L_forward) |
| Anchor | KL on retain | NLL on retain |
| Anchor strength | weak (KL≈0 at start) | strong (NLL=baseline at start) |
| Outcome | full collapse | partial collapse |

---

## 3. NPO (Negative Preference Optimization)

From Zhang et al. 2024, arXiv:2404.05868.

```
L_NPO = -(2/beta) * log_sigmoid(-beta * (log p_theta(y|x) - log p_ref(y|x)))
L     = L_NPO + alpha * L_retain_NLL
```

The sigmoid cap means NPO degrades exponentially slower than GA. `beta=0.1` is the community-standard TOFU default (OpenUnlearning `NPO.yaml`). Larger `beta` → approaches GA (more aggressive forgetting + collapse). Smaller → weak forgetting.

**Note:** Requires a frozen `ref_model` (log-ratio) — doubles ~16 GB GPU memory footprint.

---

## 4. VDU L2-anchor

`L = -L_forward + gamma * ||theta - theta_0||^2`

Simplified Heng & Soh (Selective Amnesia) style — their original paper is for conditional diffusion models, not LLMs. No TOFU-standard gamma exists; we start with `1e-3` and sweep if needed.

**Failure mode:** Pure L2 is too coarse; gamma is extremely brittle. Expected to underperform NPO.

---

## 5. Fisher-Meta

Composite of three components:

1. **EWC** (Kirkpatrick et al. 2017): `lambda * sum F_ii * (theta_i - theta_0,i)^2`, where `F_ii` is the Fisher diagonal.
2. **SalUn** (Fan et al. ICLR 2024): gradient mask — update only the top-k% most salient-on-forget weights. Default 50%.
3. **Meta-unlearning** hooks (currently stubbed): inner-loop relearn simulation.

Fisher is precomputed on retain data in `pre_unlearn_setup`. Saliency on forget data. Mask = "high saliency AND low Fisher" ⇒ edit weights that matter for forget but not for retain.

**Cost:** Pre-computation adds ~1-2 minutes on forget01. Additional memory for Fisher + saliency + mask: ~3× parameter size.

---

## 6. Exclusive Unlearning (EU)

From Sasaki et al. NLP 2026, arXiv:2604.06154.

```
L_forget = KL(p_theta || uniform_vocab)      # drive to uniform
L_retain = -E_retain[log p_theta(x)]         # standard NLL
L        = lambda * L_forget + (1 - lambda) * L_retain
```

Bounded forget loss (max = `log V`). Paper reports `lambda=0.6` as best for Llama-3.2-1B/3B-Instruct and OLMo-2-7B-Instruct across Med/Math retain domains. Insensitive to `lambda` over [0.2, 0.8]. Does **not** require a reference model.

**Paper-reported limitation:** EU-trained models are not robust to subsequent fine-tuning — 400 Alpaca examples raise ASR to ~40% (§5.4). This is directly the *relearn reversibility* question we're studying.

---

## Cross-method notes

### Reference model

KL and NPO instantiate a second frozen copy of the model on the **same GPU**. For 8B bf16, this roughly doubles memory use (~35 GB) before optimizer states. Combined with AdamW states (~32 GB) and activations, peak is ~90-95 GB on the 98 GB Blackwell GPUs.

### Retain data

All methods with `needs_retain_data()=True` cycle through `benchmark.retain_loader` in parallel with the forget loader. For TOFU retain99 (3960 samples), this never exhausts within a 10-epoch forget01 run.

### Effective batch size

Configs use `batch_size=1` + `gradient_accumulation_steps=4` ⇒ effective batch 4, not the TOFU-standard 32, to stay within memory. Revisit if results suggest batch-size-dependent behavior.
