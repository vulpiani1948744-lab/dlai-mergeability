r"""The merge-and-measure experiment.

For one (backbone, regime, seed) this

  1. reconstructs each fine-tuned encoder as $\theta_0 + \tau_i$ and records its
     accuracy on its own task -- the denominator of normalized accuracy;
  2. extracts probe features once per encoder, so every pairwise CKA afterwards
     is matrix arithmetic rather than another forward pass;
  3. for every subset of tasks and every merging algorithm, merges, evaluates,
     and writes one row carrying both the outcome and the pre-merge signals.

The pre-merge signals are computed from the task vectors of the subset alone.
Nothing in them can see the merged model, which is the property that makes them
predictions rather than descriptions.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .data import TaskData
from .evaluation import test_accuracy
from .merging import METHODS
from .models import BackboneSpec, StateDict, TaskModel, apply_task_vector
from .probe import extract_features
from .similarity import linear_cka, pairwise_summary

# Hyperparameters swept per algorithm. Task arithmetic gets a lambda sweep
# because its performance depends strongly on it and quoting one arbitrary
# value would compare our tuning against someone else's; DARE gets several
# seeds because its drop is random.
METHOD_GRID: dict[str, list[dict]] = {
    "averaging": [{}],
    "task_arithmetic": [{"scaling": s} for s in (0.1, 0.2, 0.3, 0.5, 1.0)],
    "ties": [{"density": 0.2, "scaling": s} for s in (0.5, 1.0)],
    "dare": [{"drop_rate": 0.9, "density": 0.2, "scaling": 1.0, "seed": s} for s in (0, 1, 2)],
}


@dataclass
class Bundle:
    """Everything needed to merge and evaluate one (regime, seed) group."""

    backbone: str
    regime: str
    seed: int
    spec: BackboneSpec
    theta0: StateDict                    # full state, buffers included
    taus: dict[str, StateDict]           # task name -> task vector (params only)
    heads: dict[str, nn.Linear]
    tasks: dict[str, TaskData]

    @property
    def task_names(self) -> list[str]:
        return sorted(self.taus)


def load_bundle(
    ckpt_root: Path,
    backbone: str,
    regime: str,
    seed: int,
    spec: BackboneSpec,
    tasks: list[TaskData],
) -> Bundle:
    base = Path(ckpt_root) / backbone
    theta0 = torch.load(base / "pretrained.pt", map_location="cpu")
    group = base / regime / f"seed{seed}"

    taus, heads = {}, {}
    for task in tasks:
        tau = torch.load(group / f"{task.name}_tau.pt", map_location="cpu")
        taus[task.name] = {k: v.float() for k, v in tau.items()}  # stored as fp16

        state = torch.load(group / f"{task.name}_head.pt", map_location="cpu")
        head = nn.Linear(spec.feature_dim, task.num_classes)
        head.load_state_dict(state)
        head.requires_grad_(False)
        heads[task.name] = head

    return Bundle(
        backbone=backbone, regime=regime, seed=seed, spec=spec, theta0=theta0,
        taus=taus, heads=heads, tasks={t.name: t for t in tasks},
    )


def _encoder_from(encoder: nn.Module, theta0: StateDict, tau: StateDict | None) -> nn.Module:
    """Load theta_0 (+ tau) into an existing module, in place.

    ``tau`` covers learnable parameters only; buffers come from theta_0. For a
    ResNet that means the merged model keeps the pretrained BatchNorm
    statistics, which is the standard convention and is consistent with those
    statistics having been frozen throughout fine-tuning.
    """
    state = theta0 if tau is None else apply_task_vector(theta0, tau)
    encoder.load_state_dict(state)
    return encoder


def subsets_of(names: list[str], sizes: tuple[int, ...]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for size in sizes:
        out.extend(itertools.combinations(names, size))
    return out


@torch.no_grad()
def reference_accuracies(
    bundle: Bundle, encoder: nn.Module, device: torch.device, batch_size: int
) -> dict[str, float]:
    """Accuracy of each individually fine-tuned model on its own task.

    Recomputed here rather than read from the fine-tuning logs, so this script
    is self-contained and so the denominator of normalized accuracy is measured
    with exactly the same code path as the numerator.
    """
    out = {}
    for name in bundle.task_names:
        enc = _encoder_from(encoder, bundle.theta0, bundle.taus[name]).to(device)
        model = TaskModel(enc, bundle.heads[name]).to(device)
        out[name] = test_accuracy(model, bundle.tasks[name], bundle.spec, device, batch_size)
    return out


@torch.no_grad()
def probe_features(
    bundle: Bundle,
    encoder: nn.Module,
    probe_images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Probe-set features for theta_0 and for every fine-tuned encoder.

    Extracted once and cached: with these in hand every pairwise CKA in the
    study is a matrix product, so the representation-space signals cost one
    forward pass per model rather than one per comparison.
    """
    feats = {
        "__pretrained__": extract_features(
            _encoder_from(encoder, bundle.theta0, None).to(device),
            probe_images, bundle.spec, device, batch_size,
        )
    }
    for name in bundle.task_names:
        enc = _encoder_from(encoder, bundle.theta0, bundle.taus[name]).to(device)
        feats[name] = extract_features(enc, probe_images, bundle.spec, device, batch_size)
    return feats


def signals_for(
    bundle: Bundle, subset: tuple[str, ...], feats: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Every pre-merge signal for one subset of tasks."""
    taus = [bundle.taus[n] for n in subset]
    out = pairwise_summary(taus)

    pairs = [(a, b) for i, a in enumerate(subset) for b in subset[i + 1:]]
    out["cka"] = sum(linear_cka(feats[a], feats[b]) for a, b in pairs) / len(pairs)
    out["drift"] = sum(
        linear_cka(feats["__pretrained__"], feats[n]) for n in subset
    ) / len(subset)
    return out


@torch.no_grad()
def evaluate_merge(
    bundle: Bundle,
    encoder: nn.Module,
    subset: tuple[str, ...],
    merged_tau: StateDict,
    reference: dict[str, float],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Accuracy of one merged model, absolute and normalized."""
    enc = _encoder_from(encoder, bundle.theta0, merged_tau).to(device)
    accs = {}
    for name in subset:
        model = TaskModel(enc, bundle.heads[name]).to(device)
        accs[name] = test_accuracy(model, bundle.tasks[name], bundle.spec, device, batch_size)
    return {
        "absolute_acc": sum(accs.values()) / len(accs),
        "normalized_acc": sum(accs[n] / reference[n] for n in subset) / len(subset),
    }


def run_group(
    bundle: Bundle,
    encoder: nn.Module,
    probe_images: torch.Tensor,
    device: torch.device,
    sizes: tuple[int, ...] = (2, 3, 5),
    batch_size: int = 256,
    verbose: bool = True,
) -> list[dict]:
    """Every subset x every algorithm for one (regime, seed) group."""
    reference = reference_accuracies(bundle, encoder, device, batch_size)
    if verbose:
        ref = "  ".join(f"{k.split('_')[0]} {v * 100:.1f}%" for k, v in reference.items())
        print(f"    reference accuracies: {ref}", flush=True)

    feats = probe_features(bundle, encoder, probe_images, device, batch_size)

    rows: list[dict] = []
    for subset in subsets_of(bundle.task_names, sizes):
        signals = signals_for(bundle, subset, feats)
        taus = [bundle.taus[n] for n in subset]

        for method, grid in METHOD_GRID.items():
            for params in grid:
                merged = METHODS[method](taus, **params)
                result = evaluate_merge(
                    bundle, encoder, subset, merged, reference, device, batch_size
                )
                rows.append({
                    "backbone": bundle.backbone,
                    "regime": bundle.regime,
                    "seed": bundle.seed,
                    "subset": "+".join(subset),
                    "n_tasks": len(subset),
                    "method": method,
                    **{f"param_{k}": v for k, v in params.items()},
                    **result,
                    **signals,
                })
        if verbose:
            best = max(r["normalized_acc"] for r in rows[-sum(len(g) for g in METHOD_GRID.values()):])
            print(f"    {'+'.join(subset):<60} best normalized {best * 100:.1f}%", flush=True)
    return rows
