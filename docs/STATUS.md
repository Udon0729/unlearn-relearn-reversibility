# Project status

**Last updated:** 2026-04-13

## High-level state

| Phase | State | Notes |
|---|---|---|
| Repo init, pyproject, lint/format | ✓ done | — |
| Model wrappers (LLaDA + LLaMA) | ✓ done | LLaMA primary; LLaDA kept but deprioritized ([ADR-003](adr/003-primary-axis-arm-only.md)) |
| TOFU data loader | ✓ done | response_mask includes padding ([ADR-006](adr/006-response-mask-includes-padding.md)) |
| LLaDA SFT loss correctness | ✓ fixed | [ADR-001](adr/001-llada-sft-loss-normalization.md) |
| Unlearning method corrections | ✓ done | KL on retain, NPO + retain NLL ([ADR-002](adr/002-unlearning-method-corrections.md)) |
| Fine-tune LLaMA on TOFU full | ✓ done | Verified 100% accuracy on forget01 generation |
| Phase 1: 6 unlearning methods sweep | ✓ done | See results below |
| Phase 1b: KL variants ablation (kl, grad_diff, kl_double) | ✓ done | KL fix did not help; grad_diff competitive |
| Phase 2: relearn pipeline | ⏳ next | Code exists; configs/sweep needed |
| Phase 3: cycle (unlearn→relearn × N) | ⏳ planned | — |

## Latest results (2026-04-13)

### Pre-unlearn baseline (after TOFU fine-tune)

- forget01 loss: 0.059 (ppl 1.06) — fully memorized
- utility loss: 2.211 (ppl 9.13)
- MIA AUC (forget vs test): 0.982

### Unlearning sweep on Llama-3.1-8B-Instruct, TOFU forget01

| Method | epochs | forget loss | utility loss | utility ppl | MIA AUC | verdict |
|---|---|---|---|---|---|---|
| EU (λ=0.6) | 10 | 1.18 | 1.05 | **2.85** | **0.283** | **★ 最良 (forget+utility両立)** |
| NPO | 10 | 44.3 | 2.63 | **13.8** | 0.000 | **✓ utility維持** |
| Fisher-Meta | 10 | 19.1 | 5.35 | 211 | 0.037 | △ 中程度の劣化 |
| GradDiff | 10 | 121.0 | 5.03 | 152.8 | 0.000 | △ utility劣化 |
| KL_double | 10 | 1.24 | 5.27 | 193.8 | **0.942** | × forgetが弱い (MIA高い) |
| GA | 5 | 76.8 | 13.3 | 5.7e5 | 0.001 | utility崩壊 |
| KL (修正後) | 10 | 129.1 | 82.5 | overflow | 0.000 | 完全崩壊 |
| VDU (γ=1e-3) | 10 | 142.3 | 124.6 | overflow | 0.000 | 完全崩壊 |

**主要観察：**

- **EU が最良** — forget loss 0.06→1.18（20倍）、utility ppl 9.13→**2.85（改善！）**、MIA AUC 0.98→0.28（理想0.5に近い）。論文通りλ=0.6が効果的
- **NPOも良好** — utility ppl 9.13→13.8（軽微）、forget loss 0.06→44.3（大幅）。bounded lossの恩恵
- **Fisher-Meta** — 中間的。utility ppl 211 はやや劣化、forget 19.1 で十分な忘却
- **GradDiff (TOFU標準)** — forget強いがutility劣化中程度。retain_alpha=1.0 でも GA に近い崩壊軌道
- **KL_double** — forget が弱い（MIA AUC=0.94で全く忘れていない）。KL(retain) 項が強すぎて GA 側を抑制
- **修正版 KL** — response_mask 対応にしても結局崩壊。`-L_forget` の unbounded 性が支配的、α=1.0 では止められない
- **GA/KL/VDU** — 想定通り完全崩壊（TOFU文献の baseline 結果と一致）

### KL系 3手法の比較から得られた知見

| 観点 | KL Min (`kl`) | KL Double (`kl_double`) | GradDiff (`grad_diff`) |
|---|---|---|---|
| Forget項 | -L_forget (unbounded) | -KL(forget, ref) (bounded by `log V`) | -L_forget (unbounded) |
| Anchor項 | KL(retain, ref) | KL(retain, ref) | L_retain_NLL |
| 結果 | 崩壊 | forget弱い | forget強・utility劣化 |
| 解釈 | KL anchorが弱い | bounded forget項では押せない | NLL anchorはKLより強いがGAを止めきれない |

**EUとの違い:** EUは forget項を `KL(model, uniform)` で `log V` にbound。KL_doubleとは違い、ref ではなく一様分布なので「忘却の方向」が明確で、retain項とのバランスが取りやすい。

## Next steps (priority order)

1. **Relearn pipeline configs**
   - 6 methods × relearn 1 cycle. Measure: forget recovery (loss 戻り), utility 改善, KL(p_relearn ‖ p_original)
   - 主に NPO/EU の relearn 挙動が興味深い（unlearning が成功した手法）

2. **Cycle experiments**
   - 3-5 cycles で degradation の蓄積を観察
   - cycle ごとの KL divergence 推移、relearn 収束速度の変化

3. **Forget set サイズ ablation**
   - forget01 (40) → forget05 (200) → forget10 (400)
   - 可逆性の相転移点を探る

4. **(オプション) MDM 比較追加**
   - LLaMA で十分豊かな現象が見えれば不要
   - LLaDA 1.5 のチェックポイントが残っていれば再利用可能

## Repo locations of artifacts

| Artifact | Path |
|---|---|
| Fine-tuned LLaMA checkpoint | `workdir/finetune/llama-3.1-8b-instruct_tofu_ft.pt` (~15 GB) |
| Fine-tuned LLaDA-8B-Instruct (ELBO/NLL, deprecated) | `workdir/finetune/llada-8b-instruct_*.pt` |
| Unlearning results | `results/tofu/llama-3.1-8b-instruct/{ga,kl,kl_double,grad_diff,npo,vdu,fisher_meta,eu}.json` |
| Baseline (pre-FT) | `results/baselines/tofu_*.json` (古い、参考程度) |
| Verify (FT-then-generate) | `results/finetune_verify/llama.json` (FT後 100% accuracy) |
| Logs | `logs/{finetune,unlearn}_*.log` |

## Known issues / caveats

- `torch.quantile` cannot handle tensors > ~16M elements. Use `torch.kthvalue` (already done in fisher_meta.py).
- Fisher-Meta needs CPU offload for auxiliary tensors on 8B models — see [`fisher_meta.py`](../src/unlearn_relearn/unlearning/fisher_meta.py). Slower per-step due to CPU↔GPU transfers but stable.
- Default `--config` save path is **relative to CWD**, so absolute paths in YAML `save_dir`/`output_dir` are required when launching from a different directory than the repo root.
- `transformers>=5.0` breaks LLaDA's `trust_remote_code`; pinned to `>=4.50,<4.52`.
- PyTorch nightly cu128 required for Blackwell GPU (`sm_120`). Stable 2.6 cu124 fails silently or with `sm_50..sm_90 not compatible`.
