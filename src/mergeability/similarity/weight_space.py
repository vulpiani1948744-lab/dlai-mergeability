r"""Pre-merge signals computed from task vectors alone.

These are the cheap diagnostics the project is testing. All of them read only
the checkpoints -- no data, no forward pass, no merged model -- which is the
whole point: a diagnostic you can only compute after merging and evaluating is
useless, because by then you already know the answer.

Each one is paired in the report with the merging algorithm it is predicted to
matter for (see ``merging.methods``).
"""
from __future__ import annotations

import torch

StateDict = dict[str, torch.Tensor]


def flatten(tau: StateDict) -> torch.Tensor:
    """Concatenate a task vector into one 1-D float tensor, fixed key order."""
    return torch.cat([tau[k].float().reshape(-1) for k in sorted(tau)])


# --------------------------------------------------------------------------
# b2 -- the trivial baseline
# --------------------------------------------------------------------------

def tau_norm(tau: StateDict) -> float:
    r"""$\lVert \tau \rVert_2$ -- how far fine-tuning moved the weights.

    This is baseline ``b2``, not a predictor. Any signal below that fails to
    beat it is measuring distance travelled with extra steps, and the report
    says so. Note that AdamW normalises gradient magnitudes, so with a fixed
    schedule and step count this tends to come out nearly constant across
    tasks -- in which case it has no variance and predicts nothing, which is
    itself worth reporting.
    """
    return flatten(tau).norm().item()


# --------------------------------------------------------------------------
# p1 -- task-vector alignment
# --------------------------------------------------------------------------

def cosine_similarity(tau_a: StateDict, tau_b: StateDict) -> float:
    r"""$\cos(\tau_a, \tau_b)$ over the flattened task vectors.

    The most direct reading of "do these two models want to move the weights in
    the same direction". Predicted to matter for task arithmetic, which simply
    adds the vectors up and so lets opposed directions cancel.
    """
    a, b = flatten(tau_a), flatten(tau_b)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# --------------------------------------------------------------------------
# p2 -- sign conflict
# --------------------------------------------------------------------------

def sign_conflict_rate(
    tau_a: StateDict, tau_b: StateDict, weighted: bool = True
) -> float:
    r"""Fraction of coordinates where the two task vectors disagree in sign.

    Distinct from cosine similarity, which a handful of large coordinates can
    dominate: this counts coordinates. With ``weighted=True`` each coordinate
    contributes $|\tau_{a,i}\tau_{b,i}|$, so disagreements on parameters neither
    model cares about do not drown out the ones that matter.

    Predicted to matter for TIES, whose whole mechanism is electing one sign per
    parameter and discarding the minority.
    """
    a, b = flatten(tau_a), flatten(tau_b)
    disagree = torch.sign(a) != torch.sign(b)
    if not weighted:
        return disagree.float().mean().item()
    w = (a * b).abs()
    total = w.sum()
    if total == 0:
        return 0.0
    return (w[disagree].sum() / total).item()


# --------------------------------------------------------------------------
# p3 -- singular-subspace overlap
# --------------------------------------------------------------------------

def subspace_overlap(tau_a: StateDict, tau_b: StateDict, k: int = 8) -> float:
    r"""Mean cosine of the principal angles between top-$k$ singular subspaces.

    For every 2-D parameter we take the top-$k$ left singular vectors of each
    task vector's update and measure how much the two subspaces overlap: the
    singular values of $U_a^\top U_b$ are the cosines of the principal angles
    between them. 1 means the two updates live in the same low-dimensional
    subspace, 0 means orthogonal ones.

    This is the same object the course covers as PCA (12-pcavae) applied to
    weight updates rather than to data. Predicted to matter where structural
    interference dominates -- two tasks competing for the same directions.

    Layers are averaged weighted by parameter count, so a 768x768 block counts
    for more than a 192-dimensional one.
    """
    total, weight = 0.0, 0.0
    for key in sorted(tau_a):
        a, b = tau_a[key].float(), tau_b[key].float()
        if a.ndim < 2:
            continue  # biases and norm scales carry no subspace
        a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        rank = min(k, *a.shape)
        if rank < 1:
            continue
        ua = torch.linalg.svd(a, full_matrices=False).U[:, :rank]
        ub = torch.linalg.svd(b, full_matrices=False).U[:, :rank]
        cosines = torch.linalg.svdvals(ua.T @ ub).clamp(0.0, 1.0)
        n = a.numel()
        total += cosines.mean().item() * n
        weight += n
    return total / weight if weight else float("nan")


# --------------------------------------------------------------------------
# appendix -- how concentrated an update is
# --------------------------------------------------------------------------

def effective_rank(tau: StateDict, eps: float = 1e-12) -> float:
    r"""Spectral entropy of the update, $\exp(-\sum_i p_i \log p_i)$.

    With $p_i$ the singular values normalised to sum to one. Reads as "how many
    directions is this update really using": close to 1 when the update is
    essentially rank-one, large when it is spread out. Predicted to matter for
    DARE, which sparsifies at random and so should hurt concentrated updates
    more than diffuse ones.
    """
    total, weight = 0.0, 0.0
    for key in sorted(tau):
        a = tau[key].float()
        if a.ndim < 2:
            continue
        a = a.reshape(a.shape[0], -1)
        s = torch.linalg.svdvals(a)
        p = s / s.sum().clamp(min=eps)
        entropy = -(p * (p + eps).log()).sum()
        n = a.numel()
        total += entropy.exp().item() * n
        weight += n
    return total / weight if weight else float("nan")


# --------------------------------------------------------------------------
# aggregation over a set of tasks
# --------------------------------------------------------------------------

PAIRWISE = {
    "cosine": cosine_similarity,
    "sign_conflict": sign_conflict_rate,
    "subspace_overlap": subspace_overlap,
}


def pairwise_summary(taus: list[StateDict]) -> dict[str, float]:
    """Mean over all unordered pairs, for each pairwise signal.

    A merge of $T$ tasks has $\\binom{T}{2}$ pairs; the merged model sees all of
    them at once, so the summary the predictor consumes is the average.
    """
    out: dict[str, float] = {}
    for name, fn in PAIRWISE.items():
        vals = [
            fn(taus[i], taus[j])
            for i in range(len(taus))
            for j in range(i + 1, len(taus))
        ]
        out[name] = sum(vals) / len(vals) if vals else float("nan")
    out["mean_tau_norm"] = sum(tau_norm(t) for t in taus) / len(taus)
    out["num_tasks"] = float(len(taus))
    return out
