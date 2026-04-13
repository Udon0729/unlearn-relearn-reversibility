"""Single unlearning run pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from unlearn_relearn.config import ExperimentConfig
from unlearn_relearn.data import BenchmarkData, load_benchmark
from unlearn_relearn.evaluation.metrics import _unpack_batch
from unlearn_relearn.evaluation.runner import evaluate_all
from unlearn_relearn.models import load_model
from unlearn_relearn.models.base import ModelWrapper
from unlearn_relearn.unlearning import create_method


def _load_ft_ckpt_path(cfg: ExperimentConfig) -> Path | None:
    """Return the expected fine-tuned checkpoint path, or None if not enabled."""
    if not cfg.finetune.enabled:
        return None
    loss_mode = f"_{cfg.loss.mode}" if cfg.model.type == "mdm" else ""
    return Path(cfg.finetune.save_dir) / f"{cfg.model.name}{loss_mode}_tofu_ft.pt"


def _make_scheduler(optimizer, total_steps: int, warmup_steps: int, name: str) -> LambdaLR:
    """Create a linear warmup + linear decay (or constant) scheduler."""
    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_unlearn(cfg: ExperimentConfig) -> tuple[ModelWrapper, dict]:
    """Execute a full unlearn pipeline: load model, unlearn, evaluate."""
    model = load_model(cfg.model)

    ckpt_path = _load_ft_ckpt_path(cfg)
    if ckpt_path is not None and ckpt_path.exists():
        print(f"Loading fine-tuned checkpoint: {ckpt_path}", flush=True)
        model.load_checkpoint(str(ckpt_path))

    benchmark = load_benchmark(cfg.benchmark, model.tokenizer, cfg.unlearning.batch_size)

    # Pre-unlearn evaluation
    pre_eval = evaluate_all(model, benchmark, cfg.loss)

    # Load reference model if needed
    ref_model = None
    method = create_method(cfg.unlearning, model)
    if method.needs_ref_model():
        ref_model = load_model(cfg.model)
        if ckpt_path is not None and ckpt_path.exists():
            ref_model.load_checkpoint(str(ckpt_path))
        ref_model.eval()
        method = create_method(cfg.unlearning, model, ref_model)

    method.pre_unlearn_setup(benchmark.forget_loader, benchmark.retain_loader)

    model, unlearn_log = _unlearn_loop(model, method, benchmark, cfg)

    post_eval = evaluate_all(model, benchmark, cfg.loss)

    results = {
        "config": {
            "model": cfg.model.name,
            "method": cfg.unlearning.method,
            "loss_mode": cfg.loss.mode,
            "benchmark": cfg.benchmark.name,
            "epochs": cfg.unlearning.epochs,
            "steps": cfg.unlearning.steps,
            "lr": cfg.unlearning.lr,
            "effective_batch_size": cfg.unlearning.batch_size
            * cfg.unlearning.gradient_accumulation_steps,
            "method_params": cfg.unlearning.method_params,
        },
        "pre_unlearn": pre_eval,
        "post_unlearn": post_eval,
        "unlearn_log": unlearn_log,
        "benchmark_metadata": benchmark.metadata,
    }

    return model, results


def run_unlearn_on_model(
    model: ModelWrapper,
    benchmark: BenchmarkData,
    cfg: ExperimentConfig,
) -> tuple[ModelWrapper, dict]:
    """Run unlearning on an already-loaded model (for cycle experiments)."""
    ref_model = None
    method = create_method(cfg.unlearning, model)
    if method.needs_ref_model():
        ref_model = load_model(cfg.model)
        ref_model.eval()
        method = create_method(cfg.unlearning, model, ref_model)

    method.pre_unlearn_setup(benchmark.forget_loader, benchmark.retain_loader)
    model, unlearn_log = _unlearn_loop(model, method, benchmark, cfg)
    post_eval = evaluate_all(model, benchmark, cfg.loss)

    return model, {"post_unlearn": post_eval, "unlearn_log": unlearn_log}


def _get_next(it, loader):
    """Return next batch from iterator, restarting if exhausted."""
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def _unlearn_loop(
    model: ModelWrapper,
    method,
    benchmark: BenchmarkData,
    cfg: ExperimentConfig,
) -> tuple[ModelWrapper, list[dict]]:
    """Core unlearning training loop with epoch + gradient accumulation support."""
    u = cfg.unlearning

    # Compute total forward steps (batches seen)
    n_per_epoch = len(benchmark.forget_loader)
    if u.epochs > 0 and u.steps == 0:
        total_fwd_steps = n_per_epoch * u.epochs
    elif u.steps > 0:
        total_fwd_steps = u.steps
    else:
        raise ValueError("Either unlearning.epochs or unlearning.steps must be > 0")

    grad_accum = max(u.gradient_accumulation_steps, 1)
    total_optim_steps = max(total_fwd_steps // grad_accum, 1)
    warmup_optim_steps = int(total_optim_steps * u.warmup_ratio)

    print(
        f"Unlearning: {total_fwd_steps} fwd steps, {total_optim_steps} optim steps, "
        f"warmup={warmup_optim_steps}, grad_accum={grad_accum}",
        flush=True,
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=u.lr, weight_decay=0.01)
    scheduler = _make_scheduler(optimizer, total_optim_steps, warmup_optim_steps, u.lr_scheduler)
    log = []

    forget_iter = iter(benchmark.forget_loader)
    retain_iter = iter(benchmark.retain_loader) if method.needs_retain_data() else None

    optimizer.zero_grad()
    optim_step_count = 0

    for fwd_step in tqdm(range(total_fwd_steps), desc="Unlearning"):
        forget_batch, forget_iter = _get_next(forget_iter, benchmark.forget_loader)
        forget_ids, forget_rm = _unpack_batch(forget_batch, model.device)

        retain_ids = None
        retain_rm = None
        if retain_iter is not None:
            retain_batch, retain_iter = _get_next(retain_iter, benchmark.retain_loader)
            retain_ids, retain_rm = _unpack_batch(retain_batch, model.device)
            method.set_retain_response_mask(retain_rm)

        method.set_forget_response_mask(forget_rm)
        fwd = model.forward_pass(forget_ids, cfg.loss, response_mask=forget_rm)
        loss = method.compute_loss(forget_ids, retain_ids, fwd)
        scaled = loss / grad_accum
        loss_val = loss.item()
        del fwd, loss
        scaled.backward()
        del scaled

        if (fwd_step + 1) % grad_accum == 0 or (fwd_step + 1) == total_fwd_steps:
            method.post_backward(model)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_step_count += 1

        log.append(
            {
                "fwd_step": fwd_step,
                "optim_step": optim_step_count,
                "loss": loss_val,
                "lr": scheduler.get_last_lr()[0],
            }
        )

    return model, log
