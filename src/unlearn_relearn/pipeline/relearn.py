"""Relearning pipeline: fine-tune unlearned model back on forget data."""

from __future__ import annotations

import torch
from tqdm import tqdm

from unlearn_relearn.config import ExperimentConfig
from unlearn_relearn.data import BenchmarkData
from unlearn_relearn.evaluation.metrics import _unpack_batch
from unlearn_relearn.evaluation.runner import evaluate_all
from unlearn_relearn.models.base import ModelWrapper


def run_relearn(
    model: ModelWrapper,
    benchmark: BenchmarkData,
    cfg: ExperimentConfig,
) -> dict:
    """Fine-tune the unlearned model back on forget data.

    Tracks loss curve and evaluates at regular intervals.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.relearn.lr)

    forget_iter = iter(benchmark.forget_loader)
    loss_curve: list[float] = []
    eval_log: list[dict] = []
    eval_interval = max(cfg.relearn.max_steps // 10, 1)

    for step in tqdm(range(cfg.relearn.max_steps), desc="Relearning"):
        try:
            batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(benchmark.forget_loader)
            batch = next(forget_iter)

        input_ids, response_mask = _unpack_batch(batch, model.device)
        fwd = model.forward_pass(input_ids, cfg.loss, response_mask=response_mask)
        fwd.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_curve.append(fwd.loss.item())

        # Periodic evaluation
        if (step + 1) % eval_interval == 0 or step == cfg.relearn.max_steps - 1:
            eval_results = evaluate_all(model, benchmark, cfg.loss)
            eval_log.append({"step": step, **eval_results})

    from unlearn_relearn.evaluation.metrics import relearn_convergence

    return {
        "loss_curve": loss_curve,
        "convergence": relearn_convergence(loss_curve),
        "eval_log": eval_log,
    }
