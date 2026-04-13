"""Compute baseline metrics (no unlearning) for a given config."""
import json
import sys
import torch
from pathlib import Path

from unlearn_relearn.config import ExperimentConfig
from unlearn_relearn.models import load_model
from unlearn_relearn.data import load_benchmark
from unlearn_relearn.evaluation.runner import evaluate_all

config_path = sys.argv[1]
output_path = sys.argv[2]

cfg = ExperimentConfig.from_yaml(config_path)
print(f"Baseline: {cfg.model.name}, loss={cfg.loss.mode}, benchmark={cfg.benchmark.name}", flush=True)

model = load_model(cfg.model)
bm = load_benchmark(cfg.benchmark, model.tokenizer, batch_size=cfg.unlearning.batch_size)

print("Evaluating...", flush=True)
results = evaluate_all(model, bm, cfg.loss)
results["config"] = {
    "model": cfg.model.name,
    "loss_mode": cfg.loss.mode,
    "benchmark": cfg.benchmark.name,
}
results["benchmark_metadata"] = bm.metadata

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved to {output_path}", flush=True)
print(json.dumps(results, indent=2, default=str), flush=True)
