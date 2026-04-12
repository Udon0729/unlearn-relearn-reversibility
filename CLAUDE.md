# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fact-level unlearning and relearning reversibility study comparing LLaDA-8B-Instruct (masked diffusion LM) and Llama-3.1-8B-Instruct (autoregressive LM). Uses TOFU and MUSE benchmarks. Key research question: after unlearning a fact, does relearning it restore the original model distribution or cause irreversible degradation?

Builds on findings from [mdm-unlearning](https://github.com/Udon0729/mdm-unlearning) which showed fact-level unlearning succeeds (selectivity -0.663) while corpus-level fails for both ARM and MDM.

## Build and development commands

```sh
uv pip install -e ".[viz,dev]"
uv pip install flash-attn --no-build-isolation   # separate, must match CUDA 12

ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest
pytest tests/test_foo.py   # single file
pre-commit run --all-files
```

## Running experiments

```sh
# Single experiment (config-driven)
python -m llada_unlearn.run --config configs/tofu_llada_elbo.yaml

# With relearn cycle
python -m llada_unlearn.run --config configs/tofu_llada_elbo.yaml \
    --override relearn.enabled=true relearn.num_cycles=3

# CLI override
python -m llada_unlearn.run --config configs/tofu_llama.yaml \
    --override unlearning.method=eu unlearning.steps=1000
```

## Architecture

### Source layout (`src/llada_unlearn/`)

- **`config.py`** — Dataclass config hierarchy loaded from YAML. `ExperimentConfig.from_yaml()` with CLI override support.
- **`models/`** — `ModelWrapper` ABC with `LLaDAWrapper` (ELBO + NLL dual mode) and `LLaMAWrapper` (causal LM). HuggingFace model loading via `accelerate`.
- **`data/`** — TOFU and MUSE benchmark loaders returning `BenchmarkData` (forget/retain/test DataLoaders).
- **`unlearning/`** — Six methods (GA, KL, NPO, VDU, Fisher-Meta, EU) as `UnlearningMethod` subclasses with `compute_loss()` interface. Registry in `__init__.py`.
- **`evaluation/`** — Metrics: forget_quality, model_utility, MIA, KL divergence, relearn convergence. `evaluate_all()` orchestrator.
- **`pipeline/`** — Three stages: `unlearn.py` (single run), `relearn.py` (convergence tracking), `cycle.py` (repeated unlearn→relearn with degradation tracking).
- **`run.py`** — CLI entry point dispatching to pipeline stages.

### Key design patterns

- Config-driven: all experiments specified via YAML + CLI overrides (no scattered argparse).
- `ModelWrapper.forward_pass()` returns `ForwardResult` (loss, logits, mask info) — unlearning methods compose on top.
- Loss ablation for LLaDA: `LossConfig.mode` switches between ELBO (random t) and NLL (t=1 fixed).
- Results are JSON files under `results/{benchmark}/{model}/`.

## Code quality

- Python 3.11+, ruff line-length 100, double-quote style
- ruff rules: E, W, F, I, B, C4, UP, SIM, RUF
- mypy with `check_untyped_defs = true`, `ignore_missing_imports = true`
