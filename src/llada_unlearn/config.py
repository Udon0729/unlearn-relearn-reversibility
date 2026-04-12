"""Experiment configuration: dataclass hierarchy + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from dacite import from_dict


@dataclass
class ModelConfig:
    name: str  # "llada-8b-instruct" | "llama-3.1-8b-instruct"
    type: Literal["mdm", "arm"]
    hf_id: str  # "GSAI-ML/LLaDA-8B-Instruct" etc.
    dtype: str = "bfloat16"


@dataclass
class LossConfig:
    mode: Literal["elbo", "nll"] = "elbo"  # ablation axis (ignored for ARM)
    mc_samples: int = 1  # MC samples for ELBO estimation


@dataclass
class UnlearningConfig:
    method: str = "eu"  # ga/kl/npo/vdu/fisher_meta/eu
    lr: float = 5e-5
    steps: int = 500
    batch_size: int = 1
    method_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    name: Literal["tofu", "muse"] = "tofu"
    forget_split: str = "forget01"
    retain_split: str = "retain99"


@dataclass
class RelearnConfig:
    enabled: bool = False
    lr: float = 5e-5
    max_steps: int = 500
    num_cycles: int = 1


@dataclass
class ExperimentConfig:
    model: ModelConfig
    loss: LossConfig = field(default_factory=LossConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    relearn: RelearnConfig = field(default_factory=RelearnConfig)
    output_dir: str = "results"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
        """Load config from YAML file with optional 'key.subkey=value' overrides."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        if overrides:
            for override in overrides:
                key, value = override.split("=", 1)
                parts = key.split(".")
                d = raw
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                # Auto-cast common types
                d[parts[-1]] = _cast_value(value)

        return from_dict(data_class=cls, data=raw)


def _cast_value(value: str) -> Any:
    """Cast string value to appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
