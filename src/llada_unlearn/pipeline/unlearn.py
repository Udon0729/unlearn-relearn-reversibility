"""Single unlearning run pipeline."""

from __future__ import annotations

import torch
from tqdm import tqdm

from llada_unlearn.config import ExperimentConfig
from llada_unlearn.data import BenchmarkData, load_benchmark
from llada_unlearn.evaluation.runner import evaluate_all
from llada_unlearn.models import load_model
from llada_unlearn.models.base import ModelWrapper
from llada_unlearn.unlearning import create_method


def run_unlearn(cfg: ExperimentConfig) -> tuple[ModelWrapper, dict]:
    """Execute a full unlearn pipeline: load model, unlearn, evaluate."""
    model = load_model(cfg.model)
    benchmark = load_benchmark(cfg.benchmark, model.tokenizer, cfg.unlearning.batch_size)

    # Pre-unlearn evaluation
    pre_eval = evaluate_all(model, benchmark, cfg.loss)

    # Load reference model if needed
    ref_model = None
    method = create_method(cfg.unlearning, model)
    if method.needs_ref_model():
        ref_model = load_model(cfg.model)
        ref_model.eval()
        method = create_method(cfg.unlearning, model, ref_model)

    # Pre-unlearn setup (Fisher, saliency, etc.)
    method.pre_unlearn_setup(benchmark.forget_loader, benchmark.retain_loader)

    # Unlearning loop
    model, unlearn_log = _unlearn_loop(model, method, benchmark, cfg)

    # Post-unlearn evaluation
    post_eval = evaluate_all(model, benchmark, cfg.loss)

    results = {
        "config": {
            "model": cfg.model.name,
            "method": cfg.unlearning.method,
            "loss_mode": cfg.loss.mode,
            "benchmark": cfg.benchmark.name,
            "steps": cfg.unlearning.steps,
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


def _unlearn_loop(
    model: ModelWrapper,
    method,
    benchmark: BenchmarkData,
    cfg: ExperimentConfig,
) -> tuple[ModelWrapper, list[dict]]:
    """Core unlearning training loop."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.unlearning.lr)
    log = []

    forget_iter = iter(benchmark.forget_loader)
    retain_iter = iter(benchmark.retain_loader) if method.needs_retain_data() else None

    for step in tqdm(range(cfg.unlearning.steps), desc="Unlearning"):
        # Get forget batch
        try:
            (forget_ids,) = next(forget_iter)
        except StopIteration:
            forget_iter = iter(benchmark.forget_loader)
            (forget_ids,) = next(forget_iter)
        forget_ids = forget_ids.to(model.device)

        # Get retain batch if needed
        retain_ids = None
        if retain_iter is not None:
            try:
                (retain_ids,) = next(retain_iter)
            except StopIteration:
                retain_iter = iter(benchmark.retain_loader)
                (retain_ids,) = next(retain_iter)
            retain_ids = retain_ids.to(model.device)

        # Forward + compute loss
        fwd = model.forward_pass(forget_ids, cfg.loss)
        total_loss = method.compute_loss(forget_ids, retain_ids, fwd)

        # Backward + apply masks
        total_loss.backward()
        method.post_backward(model)
        optimizer.step()
        optimizer.zero_grad()

        log.append({"step": step, "loss": total_loss.item()})

    return model, log
