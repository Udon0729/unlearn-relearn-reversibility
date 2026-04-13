# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Unlearn → Relearn reversibility study** on TOFU using Llama-3.1-8B-Instruct as the primary model. Core research question: after unlearning a fact, does relearning restore the original distribution or cause irreversible degradation?

**Current focus** (see [docs/STATUS.md](docs/STATUS.md) for live progress):
- Phase 1 (Unlearning): Compare 6 methods (GA / KL / NPO / VDU / Fisher-Meta / EU) on fine-tuned LLaMA
- Phase 2 (Relearning): Measure distribution reversibility per method (novel contribution)
- MDM (LLaDA) comparison is **secondary/optional** — see [docs/adr/](docs/adr/) for the pivot rationale

Builds on [mdm-unlearning](https://github.com/Udon0729/mdm-unlearning) which showed fact-level unlearning succeeds (selectivity -0.663) while corpus-level fails.

## Build and development commands

```sh
uv pip install -e ".[viz,dev]"
uv pip install flash-attn --no-build-isolation   # separate, must match CUDA 12

ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest
```

## Running experiments

```sh
# Step 1: Fine-tune on TOFU full (required before unlearning)
python -m unlearn_relearn.run --config configs/tofu_llama_eu.yaml --finetune-only

# Step 2: Run unlearning (any of the 6 methods)
python -m unlearn_relearn.run --config configs/tofu_llama_eu.yaml

# Step 3: With relearn cycle (planned)
python -m unlearn_relearn.run --config configs/tofu_llama_eu.yaml \
    --override relearn.enabled=true relearn.num_cycles=3
```

### GPU + process conventions (important)

- **Do not use GPU 0, 1, 2** — reserved for other users. Use GPUs 3–6 only.
- **Launch GPU jobs with `nohup`** — sessions can disconnect; nohup keeps processes alive.
- **Never combine `nohup ... &` with the Bash tool's `run_in_background: true`** — creates orphan zombies that hold GPU memory. Use one or the other.
- **Always use `python -u`** for unbuffered output (tqdm progress bars otherwise get buffered).
- Checkpoints saved under `workdir/finetune/` (absolute path in configs).

## Architecture

### Source layout (`src/unlearn_relearn/`)

- **`config.py`** — Dataclass config hierarchy (YAML loader). `UnlearningConfig` supports `epochs` (TOFU convention) or `steps`, plus `gradient_accumulation_steps`, `warmup_ratio`, `lr_scheduler` (`"linear"` or `"constant"`).
- **`models/`** — `ModelWrapper` ABC with `LLaDAWrapper` (ELBO + NLL dual mode) and `LLaMAWrapper` (causal LM, response-only loss). HuggingFace loading via `device_map={"": 0}` (never `"auto"` — causes meta-device bugs).
- **`data/`** — TOFU loader builds `response_mask` **including padding** after prompt (matches official LLaDA SFT). LLaMAWrapper also excludes padding from loss via `non_pad & response_mask`.
- **`unlearning/`** — Six methods as `UnlearningMethod` subclasses. KL uses retain samples for KL term; NPO adds retain NLL regularizer (see [docs/adr/002-unlearning-method-corrections.md](docs/adr/002-unlearning-method-corrections.md)).
- **`evaluation/`** — `forget_quality`, `model_utility`, `membership_inference` (MIA AUC), `kl_divergence`, `relearn_convergence`.
- **`pipeline/`** — `unlearn.py` (single run, epoch-based), `relearn.py`, `cycle.py`.
- **`run.py`** — CLI entry point.

### Key design patterns

- **Config-driven**: YAML + CLI overrides (`--override key.subkey=value`).
- `ModelWrapper.forward_pass()` takes optional `response_mask` for SFT-style masking.
- `UnlearningMethod.needs_ref_model()` / `needs_retain_data()` flags wire up data/ref-model loading in the pipeline.
- Results JSON under `results/{benchmark}/{model}/{method}.json`.

### Critical code facts

- **LLaDA SFT loss** (official form): `sum(token_loss / p_mask / answer_length_per_sample) / batch_size`. Early buggy version used `sum / total_tokens` which biased toward long samples. See [docs/adr/001-llada-sft-loss-normalization.md](docs/adr/001-llada-sft-loss-normalization.md).
- **response_mask** includes padding tokens after the prompt (LLaDA needs this for EOS prediction); LLaMAWrapper internally excludes padding via `non_pad & response_mask` for TOFU convention.
- **Fine-tune config standards** (all configs): `lr=1e-5`, `epochs=5`, `batch_size=1`, `grad_accum=16`, `max_length=256`, `weight_decay=0.01`. Peak GPU memory ~80GB.

## Code quality

- Python 3.11+, ruff line-length 100, double-quote style
- ruff rules: E, W, F, I, B, C4, UP, SIM, RUF
- mypy with `check_untyped_defs = true`, `ignore_missing_imports = true`

## See also

- [docs/STATUS.md](docs/STATUS.md) — live experiment status and latest results
- [docs/adr/](docs/adr/) — architecture decision records
- [docs/METHODS.md](docs/METHODS.md) — per-method hyperparameter rationale from literature
