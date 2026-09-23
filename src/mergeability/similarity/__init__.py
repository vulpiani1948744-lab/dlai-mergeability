"""Pre-merge similarity signals: weight space and representation space."""
from .cka import build_probe_set, encoder_cka, linear_cka
from .weight_space import (
    PAIRWISE,
    cosine_similarity,
    effective_rank,
    flatten,
    pairwise_summary,
    sign_conflict_rate,
    subspace_overlap,
    tau_norm,
)

__all__ = [
    "PAIRWISE",
    "build_probe_set",
    "cosine_similarity",
    "effective_rank",
    "encoder_cka",
    "flatten",
    "linear_cka",
    "pairwise_summary",
    "sign_conflict_rate",
    "subspace_overlap",
    "tau_norm",
]
