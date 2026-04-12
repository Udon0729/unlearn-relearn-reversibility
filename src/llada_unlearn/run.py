"""CLI entry point for experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llada_unlearn.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaDA Unlearn-Relearn Experiments")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values: key.subkey=value",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config, overrides=args.override)

    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Dispatch to appropriate pipeline
    if cfg.relearn.enabled and cfg.relearn.num_cycles > 1:
        from llada_unlearn.pipeline.cycle import run_cycles

        results = run_cycles(cfg)
        suffix = f"_cycles{cfg.relearn.num_cycles}"
    elif cfg.relearn.enabled:
        from llada_unlearn.pipeline.relearn import run_relearn
        from llada_unlearn.pipeline.unlearn import run_unlearn

        model, unlearn_results = run_unlearn(cfg)

        from llada_unlearn.data import load_benchmark

        benchmark = load_benchmark(cfg.benchmark, model.tokenizer, cfg.unlearning.batch_size)
        relearn_results = run_relearn(model, benchmark, cfg)
        results = {**unlearn_results, "relearn": relearn_results}
        suffix = "_relearn"
    else:
        from llada_unlearn.pipeline.unlearn import run_unlearn

        _, results = run_unlearn(cfg)
        suffix = ""

    # Save results
    output_dir = Path(cfg.output_dir) / cfg.benchmark.name / cfg.model.name
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_mode = f"_{cfg.loss.mode}" if cfg.model.type == "mdm" else ""
    filename = f"{cfg.unlearning.method}{loss_mode}{suffix}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
