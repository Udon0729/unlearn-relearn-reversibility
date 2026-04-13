# ADR-005: GPU allocation and process management conventions

**Date:** 2026-04-12
**Status:** Accepted

## Context

The shared machine has 7 × 98 GB Blackwell GPUs. GPUs 0–2 are used by other users / jobs; 3–6 are available for this project.

Early experiments suffered from multiple compounding process-management failures:
1. Combining `nohup cmd &` with the Bash tool's `run_in_background: true` caused the launcher shell to finish immediately while the orphaned Python process kept running. When it OOM'd, nothing killed it, leaving GPU memory reserved but invisible to `nvidia-smi`-free (shown as `fuser /dev/nvidia6`).
2. Output buffering in nohup hid OOM tracebacks until the process was manually terminated. `python -u` plus `2>&1 | tee` fixed this.
3. Setting `device_map="auto"` inside `from_pretrained` triggered a transformers v5 bug (`all_tied_weights_keys`) that did not appear with `device_map={"": 0}`.
4. `transformers>=5.0` broke LLaDA's `trust_remote_code` path. Pinned to `4.50..4.52`.
5. PyTorch stable (2.6 cu124) does not support Blackwell `sm_120`; nightly (`cu128`) is required.

## Decision

### GPU allocation

- **Do not use GPUs 0, 1, 2** unless explicitly told otherwise.
- Use GPUs 3–6 only.
- Before launching, always check `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` to confirm the target GPU is free.

### Process management

- **Always use `nohup`** (no `&` inside Bash-tool calls using `run_in_background`) to survive session disconnects.
- **Never combine `nohup ... &` with `run_in_background: true`** — pick one:
  - Option A: `nohup ... &` with regular Bash tool call (no `run_in_background`). Zombie-safe; survives disconnects.
  - Option B: direct `python -u ...` with `run_in_background: true`. Accurate lifecycle tracking; dies if session ends.
- **Always `python -u`** for unbuffered stdout/stderr (especially important for tqdm progress bars).
- Always redirect both stdout and stderr: `> log 2>&1`.

### Model loading

- `device_map={"": 0}` — never `"auto"` with LLaDA (custom `trust_remote_code` model).
- `torch_dtype=torch.bfloat16` for both LLaDA and LLaMA.
- `transformers>=4.50,<4.52`, `torch>=2.12.0.dev` (nightly cu128) pinned in `pyproject.toml`.

### Memory budgets (8B bf16, batch_size=1, seq_len=256)

- Model only: ~16 GB
- + AdamW states: ~48 GB
- Peak during training: ~80 GB (fits 98 GB with headroom)
- With reference model (KL, NPO): ~95 GB — works but leaves very little slack

## Consequences

- Sweeps over 6 methods use GPUs 3–6; at most 4 methods run in parallel, the remaining 2 wait.
- Fisher-Meta is noticeably slower because its `pre_unlearn_setup` computes a full Fisher diagonal on retain data.

## References

- `feedback_gpu_usage.md`, `feedback_nohup.md` in the user's memory
- PyTorch nightly install: `uv pip install --reinstall torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128`
