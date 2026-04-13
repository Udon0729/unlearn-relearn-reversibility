"""Unlearning method registry.

Available methods:
- ga: Gradient Ascent (naive baseline)
- kl: KL-Minimization (-L_forget + alpha * KL on retain) — TOFU paper
- kl_double: KL on both forget (max) and retain (min)
- grad_diff: -L_forget + alpha * L_retain_NLL — TOFU GradDiff
- npo: Negative Preference Optimization (Zhang et al. 2024)
- vdu: VDU L2-anchor
- fisher_meta: Fisher-EWC + SalUn saliency mask
- eu: Exclusive Unlearning (Sasaki et al. 2026)
"""

from __future__ import annotations

from unlearn_relearn.config import UnlearningConfig
from unlearn_relearn.models.base import ModelWrapper
from unlearn_relearn.unlearning.base import UnlearningMethod
from unlearn_relearn.unlearning.eu import ExclusiveUnlearning
from unlearn_relearn.unlearning.fisher_meta import FisherMeta
from unlearn_relearn.unlearning.ga import GradientAscent
from unlearn_relearn.unlearning.grad_diff import GradDiff
from unlearn_relearn.unlearning.kl import KLConstrained
from unlearn_relearn.unlearning.kl_double import KLDouble
from unlearn_relearn.unlearning.npo import NPO
from unlearn_relearn.unlearning.vdu import VDU

METHODS: dict[str, type[UnlearningMethod]] = {
    "ga": GradientAscent,
    "kl": KLConstrained,
    "kl_double": KLDouble,
    "grad_diff": GradDiff,
    "npo": NPO,
    "vdu": VDU,
    "fisher_meta": FisherMeta,
    "eu": ExclusiveUnlearning,
}


def create_method(
    cfg: UnlearningConfig,
    model: ModelWrapper,
    ref_model: ModelWrapper | None = None,
) -> UnlearningMethod:
    """Instantiate an unlearning method by name."""
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method '{cfg.method}'. Available: {list(METHODS)}")
    return METHODS[cfg.method](cfg, model, ref_model)


__all__ = ["METHODS", "UnlearningMethod", "create_method"]
