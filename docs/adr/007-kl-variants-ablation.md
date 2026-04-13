# ADR-007: KL系手法のablation (kl, kl_double, grad_diff)

**Date:** 2026-04-13
**Status:** Accepted

## Context

初回の8手法sweepで `kl` (TOFU "KL Minimization", `-L_forget + α·KL(retain)`) が完全崩壊（utility ppl overflow）。NPOやEUは安定していたため、原因を分析した結果：

- `-L_forget` は unbounded で急速に増大
- `α·KL(retain, ref)` は初期 KL=0 でほぼ効かず、徐々にしか効果が出ない
- `α=1.0` では GA に対して引き戻しが間に合わない

加えて、TOFU 原論文には複数の KL 系亜種が存在することを再確認:
- "Gradient Difference" = `-L_forget + α · L_retain_NLL` （NLL anchor）
- "KL Minimization" = `-L_forget + α · KL(retain, ref)` （現在の `kl`）
- 「両側 KL」(`-α_f · KL(forget) + α_r · KL(retain)`) は文献では稀だが論理的なバリエーション

## Decision

3つの手法を ablation として並列実装:

1. **`kl` (修正版)**: response_mask を retain forward に渡して、KL を応答トークン位置のみで計算（padding希釈解消）
2. **`grad_diff` (新規)**: TOFU 原論文の "Gradient Difference"。`-L_forget + α · L_retain_NLL`
3. **`kl_double` (新規)**: 両側 KL。`-α_f · KL_forget + α_r · KL_retain`

`base.py` に `set_forget_response_mask` / `set_retain_response_mask` を追加し、パイプラインから各手法に response_mask を伝達できるようにした。

## Empirical results (TOFU forget01, Llama-3.1-8B-Instruct, ε epochs=10, lr=1e-5)

| 手法 | forget loss | utility ppl | MIA AUC | 知見 |
|---|---|---|---|---|
| `kl` (修正後) | 129.1 | 6.6e35 | 0.000 | **依然崩壊**。response_mask対応では効果不足 |
| `grad_diff` | 121.0 | 152.8 | 0.000 | NLL anchorは KLより強いが GA を止めきれない |
| `kl_double` | 1.24 | 193.8 | **0.942** | **forget が全く効いていない**（MIA AUC=0.94） |

## Consequences

- **`kl` の修正は本質的に効かない**: response_mask を渡しても、unbounded な `-L_forget` 項が支配的。α を10〜100まで上げる、もしくはforget項自体をbounded形式（NPO/EU）に置き換える必要がある
- **`grad_diff` は中間的な選択肢**: ref model 不要で実装が簡単、retain NLL anchor は意味がある。が、forget項はGA そのものなので大規模崩壊リスクは残る
- **`kl_double` はforget が弱すぎる**: 両側 KL では「forget も bounded」のため、retain項とのバランス調整が困難。α_f を上げると `kl_forget` の上昇（拡散）に頭打ちが来て forget が進まない
- **EU の優位性が再確認**: forget項を `KL(model, uniform)` という独立した目標に切り替えることで、ref-based手法の弱点を回避している

## Future tuning options (not yet tried)

| 手法 | 改善案 |
|---|---|
| `kl` | α=10〜100に上げる |
| `kl_double` | α_f=10, α_r=0.1 のように比率を変える |
| `grad_diff` | retain_alpha=2.0〜10.0、または lr を下げる |

これらは relearn 実験の前処理として価値があるが、最良 (EU/NPO) との差を埋められる見込みは低い。

## References

- [TOFU paper §3.2](https://arxiv.org/abs/2401.06121)
- [`unlearning/kl.py`](../../src/unlearn_relearn/unlearning/kl.py), [`grad_diff.py`](../../src/unlearn_relearn/unlearning/grad_diff.py), [`kl_double.py`](../../src/unlearn_relearn/unlearning/kl_double.py)
- [docs/METHODS.md](../METHODS.md) — KL 系セクション
