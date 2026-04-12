"""Unlearn-relearn cycle pipeline: repeated cycles with degradation tracking."""

from __future__ import annotations

from llada_unlearn.config import ExperimentConfig
from llada_unlearn.data import load_benchmark
from llada_unlearn.evaluation.metrics import kl_divergence
from llada_unlearn.evaluation.runner import evaluate_all
from llada_unlearn.models import load_model
from llada_unlearn.pipeline.relearn import run_relearn
from llada_unlearn.pipeline.unlearn import run_unlearn_on_model


def run_cycles(cfg: ExperimentConfig) -> dict:
    """Run N unlearn-relearn cycles, tracking distribution divergence.

    At each cycle boundary, measures:
    - KL(p_current || p_original) on test data
    - Forget quality and model utility
    - Relearning convergence speed
    """
    # Load original model (reference for KL measurements)
    original_model = load_model(cfg.model)
    original_model.eval()

    # Load working model
    model = load_model(cfg.model)
    benchmark = load_benchmark(cfg.benchmark, model.tokenizer, cfg.unlearning.batch_size)

    # Baseline evaluation
    baseline_eval = evaluate_all(model, benchmark, cfg.loss)
    cycle_results = []

    for i in range(cfg.relearn.num_cycles):
        # --- Unlearn ---
        model, unlearn_res = run_unlearn_on_model(model, benchmark, cfg)

        kl_after_unlearn = kl_divergence(model, original_model, benchmark.test_loader, cfg.loss)

        # --- Relearn ---
        relearn_res = run_relearn(model, benchmark, cfg)

        kl_after_relearn = kl_divergence(model, original_model, benchmark.test_loader, cfg.loss)

        post_relearn_eval = evaluate_all(model, benchmark, cfg.loss)

        cycle_results.append(
            {
                "cycle": i,
                "unlearn": unlearn_res,
                "relearn": relearn_res,
                "kl_after_unlearn": kl_after_unlearn,
                "kl_after_relearn": kl_after_relearn,
                "post_relearn_eval": post_relearn_eval,
            }
        )

    return {
        "baseline": baseline_eval,
        "cycles": cycle_results,
        "config": {
            "model": cfg.model.name,
            "method": cfg.unlearning.method,
            "loss_mode": cfg.loss.mode,
            "benchmark": cfg.benchmark.name,
            "num_cycles": cfg.relearn.num_cycles,
        },
    }
