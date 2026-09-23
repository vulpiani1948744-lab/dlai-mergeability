r"""The four merging algorithms.

Every method takes a list of task vectors $\{\tau_1 \dots \tau_T\}$ and returns a
single combined task vector, to be applied to the pretrained weights with
``models.apply_task_vector``. None of them touches the data or runs a gradient
step; merging is arithmetic on weights.

The four are not a survey. They were chosen because they fail for structurally
different reasons, which is what makes the project's central claim testable:
if mergeability were an intrinsic property of a pair of models, one diagnostic
would predict failure for all of them.

===============  ==================================  ============================
method           mechanism it relies on              predicted failure mode
===============  ==================================  ============================
averaging        none -- unweighted mean             drift away from $\theta_0$
task arithmetic  one global scaling factor           interference between $\tau_i$
TIES             trim, elect a sign, disjoint mean   sign conflict
DARE             stochastic sparsification           task-vector density / rank
===============  ==================================  ============================

These predictions are registered here *before* the experiments run, so each of
them can genuinely turn out to be wrong.

One honest relation to state up front, because the report has to state it too:
**weight averaging is task arithmetic with** $\lambda = 1/T$. It is kept as a
separate entry because it is the canonical baseline in the literature (model
soups), not because it is an independent mechanism.
"""
from __future__ import annotations

import torch

StateDict = dict[str, torch.Tensor]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _check(taus: list[StateDict]) -> list[str]:
    """Validate the inputs and return the shared key order."""
    if not taus:
        raise ValueError("no task vectors to merge")
    keys = sorted(taus[0])
    for i, t in enumerate(taus[1:], start=1):
        if sorted(t) != keys:
            raise ValueError(f"task vector {i} has different keys from task vector 0")
        for k in keys:
            if t[k].shape != taus[0][k].shape:
                raise ValueError(f"shape mismatch for {k!r} in task vector {i}")
    return keys


def _flat(tau: StateDict, keys: list[str]) -> torch.Tensor:
    return torch.cat([tau[k].float().reshape(-1) for k in keys])


def _unflatten(vec: torch.Tensor, reference: StateDict, keys: list[str]) -> StateDict:
    out: StateDict = {}
    offset = 0
    for k in keys:
        n = reference[k].numel()
        out[k] = vec[offset : offset + n].view_as(reference[k]).clone()
        offset += n
    return out


def _trim(flat: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the ``density`` fraction of entries with the largest magnitude.

    The threshold is global over the whole task vector, as in TIES-Merging --
    not per layer. Trimming per layer would give every layer the same budget
    regardless of how much it actually moved.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    if density == 1.0:
        return flat
    k = max(1, int(round(density * flat.numel())))
    threshold = flat.abs().kthvalue(flat.numel() - k + 1).values
    return torch.where(flat.abs() >= threshold, flat, torch.zeros_like(flat))


# --------------------------------------------------------------------------
# the methods
# --------------------------------------------------------------------------

def weight_averaging(taus: list[StateDict], **_) -> StateDict:
    r"""Unweighted mean of the task vectors -- the "model soup" baseline.

    Averaging the fine-tuned *weights* is the same thing: the mean of
    $\theta_0 + \tau_i$ is $\theta_0 + \frac{1}{T}\sum_i \tau_i$.
    """
    keys = _check(taus)
    return {k: torch.stack([t[k].float() for t in taus]).mean(0) for k in keys}


def task_arithmetic(taus: list[StateDict], scaling: float = 0.3, **_) -> StateDict:
    r"""$\lambda \sum_i \tau_i$ (Ilharco et al., ICLR 2023).

    ``scaling`` is swept in the experiments rather than fixed: the method's
    performance depends strongly on it, and reporting a single arbitrary value
    would be comparing our tuning against someone else's.
    """
    keys = _check(taus)
    return {k: scaling * torch.stack([t[k].float() for t in taus]).sum(0) for k in keys}


def ties(
    taus: list[StateDict],
    density: float = 0.2,
    scaling: float = 1.0,
    **_,
) -> StateDict:
    r"""TIES-Merging: trim, elect sign, disjoint mean (Yadav et al., NeurIPS 2023).

    1. **Trim** each $\tau_i$ to its top ``density`` fraction by magnitude.
    2. **Elect a sign** per parameter: the sign of $\sum_i \tau_i$, i.e. whichever
       direction carries the greater total magnitude.
    3. **Disjoint mean**: average only the entries that agree with the elected
       sign, so a minority pulling the other way is dropped rather than allowed
       to cancel the majority out.

    Step 3 is why sign conflict is the failure mode we expect to matter here.
    """
    keys = _check(taus)
    trimmed = torch.stack([_trim(_flat(t, keys), density) for t in taus])  # (T, N)

    elected = torch.sign(trimmed.sum(dim=0))                 # (N,)
    agrees = (torch.sign(trimmed) == elected) & (trimmed != 0)
    count = agrees.sum(dim=0).clamp(min=1)
    merged = (trimmed * agrees).sum(dim=0) / count

    # parameters where every task vector was trimmed away stay at zero
    merged = torch.where(elected == 0, torch.zeros_like(merged), merged)
    return _unflatten(scaling * merged, taus[0], keys)


def dare(
    taus: list[StateDict],
    drop_rate: float = 0.9,
    scaling: float = 1.0,
    density: float = 0.2,
    base: str = "ties",
    seed: int = 0,
    **_,
) -> StateDict:
    r"""DARE: drop entries at random, rescale the survivors, then merge.

    Each $\tau_i$ independently keeps a random $(1 - p)$ fraction of its entries
    and rescales them by $1/(1-p)$, which leaves the expected value of the task
    vector unchanged while making the surviving supports of different tasks
    mostly disjoint (Yu et al., ICML 2024).

    Because the drop is random, this is the one method whose output depends on a
    seed. The seed is an explicit argument so runs stay reproducible, and the
    experiments average over several of them.
    """
    keys = _check(taus)
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

    generator = torch.Generator().manual_seed(seed)
    survived: list[StateDict] = []
    for t in taus:
        flat = _flat(t, keys)
        keep = torch.rand(flat.shape, generator=generator) >= drop_rate
        survived.append(_unflatten(flat * keep / (1.0 - drop_rate), taus[0], keys))

    if base == "ties":
        return ties(survived, density=density, scaling=scaling)
    if base == "task_arithmetic":
        return task_arithmetic(survived, scaling=scaling)
    raise ValueError(f"unknown base merger {base!r}")


METHODS = {
    "averaging": weight_averaging,
    "task_arithmetic": task_arithmetic,
    "ties": ties,
    "dare": dare,
}
