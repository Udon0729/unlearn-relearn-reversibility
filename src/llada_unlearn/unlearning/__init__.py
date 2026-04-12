"""Unlearning method registry.

Six methods ported from mdm-unlearning (untrac_mdm.py:674-732):
- ga: Gradient Ascent
- kl: KL-constrained Gradient Ascent
- npo: Negative Preference Optimization
- vdu: VDU L2-anchor
- fisher_meta: Fisher-EWC + saliency mask + meta-unlearning
- eu: Exclusive Unlearning
"""

from __future__ import annotations

from llada_unlearn.config import UnlearningConfig
from llada_unlearn.models.base import ModelWrapper
from llada_unlearn.unlearning.base import UnlearningMethod
from llada_unlearn.unlearning.eu import ExclusiveUnlearning
from llada_unlearn.unlearning.fisher_meta import FisherMeta
from llada_unlearn.unlearning.ga import GradientAscent
from llada_unlearn.unlearning.kl import KLConstrained
from llada_unlearn.unlearning.npo import NPO
from llada_unlearn.unlearning.vdu import VDU

METHODS: dict[str, type[UnlearningMethod]] = {
    "ga": GradientAscent,
    "kl": KLConstrained,
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
