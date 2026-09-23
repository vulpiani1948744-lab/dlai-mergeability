#!/usr/bin/env python
"""Correctness checks for the pieces that do not need the dataset.

Run before any real experiment. A silent bug in the augmentation or in the
task-vector arithmetic would corrupt every number downstream while still
producing plausible-looking output, which is the expensive kind of bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergeability.data import _random_crop, batches, to_model_input
from mergeability.models import (
    apply_task_vector,
    encoder_param_dict,
    flatten_state_dict,
    freeze_batchnorm,
    task_vector,
)
from mergeability.merging import METHODS, dare, task_arithmetic, ties, weight_averaging
from mergeability.merging.methods import _trim
from mergeability.probe import fit_linear_probe

PASSED, FAILED = [], []


def check(name: str):
    def deco(fn):
        try:
            fn()
        except AssertionError as e:
            FAILED.append((name, str(e)))
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:  # noqa: BLE001
            FAILED.append((name, repr(e)))
            print(f"  ERROR {name}\n        {e!r}")
        else:
            PASSED.append(name)
            print(f"  ok    {name}")
        return fn

    return deco


@check("random crop preserves shape and dtype")
def _():
    x = torch.rand(16, 3, 32, 32)
    out = _random_crop(x, pad=4)
    assert out.shape == x.shape, f"shape {out.shape}"
    assert out.dtype == x.dtype


@check("random crop is per-sample, not per-batch")
def _():
    # identical images in, different crops out => offsets vary across the batch
    x = torch.arange(32 * 32, dtype=torch.float32).view(1, 1, 32, 32).repeat(64, 3, 1, 1)
    torch.manual_seed(0)
    out = _random_crop(x, pad=4)
    distinct = {tuple(o.flatten()[:8].tolist()) for o in out}
    assert len(distinct) > 1, "every sample got the same crop"


@check("random crop with pad=0 is the identity")
def _():
    x = torch.rand(8, 3, 32, 32)
    assert torch.equal(_random_crop(x, pad=0), x)


@check("to_model_input normalises to roughly zero mean")
def _():
    x = torch.randint(0, 256, (32, 3, 32, 32), dtype=torch.uint8)
    mean = std = (0.5, 0.5, 0.5)
    out = to_model_input(x, torch.device("cpu"), 64, mean, std, train=False)
    assert out.shape == (32, 3, 64, 64), f"shape {out.shape}"
    assert abs(out.mean().item()) < 0.2, f"mean {out.mean().item():.3f}"


@check("to_model_input leaves resolution alone when it already matches")
def _():
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    out = to_model_input(x, torch.device("cpu"), 32, (0, 0, 0), (1, 1, 1), train=False)
    assert torch.allclose(out, x.float() / 255.0)


@check("batches cover every sample exactly once")
def _():
    x = torch.arange(100).view(100, 1)
    y = torch.arange(100)
    seen = torch.cat([yb for _, yb in batches(x, y, 32, shuffle=True)])
    assert torch.equal(seen.sort().values, y), "not a permutation of the dataset"


@check("task vector round-trips: theta_0 + (theta_ft - theta_0) == theta_ft")
def _():
    torch.manual_seed(0)
    net_a, net_b = nn.Linear(8, 4), nn.Linear(8, 4)
    t0 = {k: v.clone() for k, v in net_a.state_dict().items()}
    tft = {k: v.clone() for k, v in net_b.state_dict().items()}
    back = apply_task_vector(t0, task_vector(tft, t0), scaling=1.0)
    for k in tft:
        assert torch.allclose(back[k], tft[k], atol=1e-6), f"{k} did not round-trip"


@check("task vector scaling by 0 returns the pretrained weights")
def _():
    torch.manual_seed(0)
    t0 = {k: v.clone() for k, v in nn.Linear(8, 4).state_dict().items()}
    tft = {k: v.clone() for k, v in nn.Linear(8, 4).state_dict().items()}
    back = apply_task_vector(t0, task_vector(tft, t0), scaling=0.0)
    for k in t0:
        assert torch.allclose(back[k], t0[k], atol=1e-6), f"{k} moved"


@check("task vector skips integer buffers")
def _():
    bn = nn.BatchNorm2d(4)
    sd = {k: v.clone() for k, v in bn.state_dict().items()}
    sd2 = {k: (v + 1 if torch.is_floating_point(v) else v + 1) for k, v in sd.items()}
    tau = task_vector(sd2, sd)
    assert "num_batches_tracked" not in tau, "integer counter leaked into the task vector"


@check("flatten_state_dict is deterministic in key order")
def _():
    sd = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    assert torch.equal(flatten_state_dict(sd), torch.tensor([1.0, 2.0]))


@check("linear probe separates linearly separable data")
def _():
    torch.manual_seed(0)
    centres = torch.randn(4, 16) * 5
    feats = torch.cat([centres[i] + 0.3 * torch.randn(64, 16) for i in range(4)])
    labels = torch.arange(4).repeat_interleave(64)
    head = fit_linear_probe(feats, labels, num_classes=4, seed=0)
    acc = (head(feats).argmax(1) == labels).float().mean().item()
    assert acc > 0.95, f"probe accuracy {acc:.3f}"
    assert not head.weight.requires_grad, "probe head was left trainable"


@check("linear probe survives badly scaled features")
def _():
    # The probe standardises internally and folds the scaler back into the
    # layer. If that arithmetic were wrong, wildly different per-feature scales
    # would break it while the well-conditioned test above still passed.
    torch.manual_seed(0)
    scales = torch.logspace(-3, 3, 16)
    centres = torch.randn(4, 16) * 5
    feats = torch.cat([centres[i] + 0.3 * torch.randn(64, 16) for i in range(4)]) * scales
    labels = torch.arange(4).repeat_interleave(64)
    head = fit_linear_probe(feats, labels, num_classes=4, seed=0)
    acc = (head(feats).argmax(1) == labels).float().mean().item()
    assert acc > 0.95, f"probe accuracy on badly scaled features {acc:.3f}"


@check("linear probe head consumes raw, unstandardised features")
def _():
    # A head that secretly expected standardised input would score near chance
    # when handed the same raw features it was fitted on.
    torch.manual_seed(0)
    feats = torch.cat([torch.randn(64, 8) + 10 * i for i in range(3)])
    labels = torch.arange(3).repeat_interleave(64)
    head = fit_linear_probe(feats, labels, num_classes=3, seed=0)
    acc = (head(feats).argmax(1) == labels).float().mean().item()
    assert acc > 0.95, f"accuracy {acc:.3f} -- scaler fold-back is wrong"


@check("param dict excludes BatchNorm running statistics")
def _():
    # Regression: including running_mean/running_var made ||tau|| read 14414 for
    # ResNet-18 against 2.7 for ViT-Tiny. They are data statistics, not weights.
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    params = encoder_param_dict(net)
    leaked = [k for k in params if "running_" in k or "num_batches" in k]
    assert not leaked, f"buffers leaked into the task vector: {leaked}"
    assert any("1.weight" in k for k in params), "BN affine weight should stay trainable"


@check("task vector over params is unaffected by drifting BN statistics")
def _():
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    before = encoder_param_dict(net)
    net.train()
    for _ in range(5):  # move the running statistics without touching any weight
        net(torch.randn(8, 3, 8, 8) * 50)
    after = encoder_param_dict(net)
    norm = torch.cat([v.reshape(-1) for v in task_vector(after, before).values()]).norm()
    assert norm.item() < 1e-6, f"||tau|| = {norm.item():.4f} but no weight was updated"


@check("freeze_batchnorm survives a later train() call")
def _():
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    bn = net[1]
    net.train()
    n = freeze_batchnorm(net)
    assert n == 1, f"froze {n} layers, expected 1"
    assert not bn.training, "BatchNorm still in training mode"
    assert net[0].training, "the rest of the network must stay in training mode"


@check("frozen BatchNorm does not move its running statistics")
def _():
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    bn = net[1]
    net.train()
    freeze_batchnorm(net)
    before = bn.running_var.clone()
    for _ in range(5):
        net(torch.randn(8, 3, 8, 8) * 50)
    assert torch.allclose(bn.running_var, before), "running statistics drifted"


# --------------------------------------------------------------------------
# merging algorithms
# --------------------------------------------------------------------------

def _taus(values):
    """Build task vectors from plain lists, one key, for readable assertions."""
    return [{"w": torch.tensor(v, dtype=torch.float32)} for v in values]


@check("averaging equals task arithmetic with lambda = 1/T")
def _():
    # Stated in the module docstring and in the report; if it ever stops being
    # true, one of the two implementations has drifted.
    taus = _taus([[1.0, -2.0, 3.0], [0.5, 1.0, -1.0], [-1.5, 0.0, 2.0]])
    a = weight_averaging(taus)["w"]
    b = task_arithmetic(taus, scaling=1 / 3)["w"]
    assert torch.allclose(a, b, atol=1e-6), f"{a} != {b}"


@check("task arithmetic with lambda=1 on a single task vector is the identity")
def _():
    tau = _taus([[1.0, -2.0, 3.0]])
    out = task_arithmetic(tau, scaling=1.0)["w"]
    assert torch.allclose(out, tau[0]["w"]), f"{out}"


@check("trim keeps exactly the top fraction by magnitude")
def _():
    flat = torch.tensor([0.1, -5.0, 0.2, 3.0, -0.3, 4.0, 0.05, -0.2, 0.15, 0.25])
    out = _trim(flat, density=0.3)
    kept = (out != 0).sum().item()
    assert kept == 3, f"kept {kept}, expected 3"
    assert set(out[out != 0].abs().tolist()) == {5.0, 4.0, 3.0}, f"kept the wrong ones: {out}"


@check("trim with density=1 changes nothing")
def _():
    flat = torch.randn(64)
    assert torch.equal(_trim(flat, 1.0), flat)


@check("TIES reduces to averaging when every sign agrees")
def _():
    taus = _taus([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a = ties(taus, density=1.0, scaling=1.0)["w"]
    b = weight_averaging(taus)["w"]
    assert torch.allclose(a, b, atol=1e-6), f"TIES {a} vs mean {b}"


@check("TIES drops the minority sign instead of letting it cancel")
def _():
    # +3, +3, -1: the sum is positive, so the elected sign is +, and only the
    # two agreeing entries are averaged -> 3.0. A plain mean would give 1.667.
    taus = _taus([[3.0], [3.0], [-1.0]])
    out = ties(taus, density=1.0, scaling=1.0)["w"].item()
    assert abs(out - 3.0) < 1e-6, f"got {out}, expected 3.0 (plain mean would be 1.667)"


@check("TIES elects the sign carrying the greater total magnitude")
def _():
    # +1, +1, -5: the negative side wins on total magnitude.
    taus = _taus([[1.0], [1.0], [-5.0]])
    out = ties(taus, density=1.0, scaling=1.0)["w"].item()
    assert abs(out - (-5.0)) < 1e-6, f"got {out}, expected -5.0"


@check("DARE with drop_rate=0 equals its base merger")
def _():
    taus = _taus([[1.0, -2.0, 3.0], [0.5, 1.0, -1.0]])
    a = dare(taus, drop_rate=0.0, base="task_arithmetic", scaling=1.0)["w"]
    b = task_arithmetic(taus, scaling=1.0)["w"]
    assert torch.allclose(a, b, atol=1e-6), f"{a} != {b}"


@check("DARE preserves the task vector in expectation")
def _():
    # Dropping p of the entries and rescaling survivors by 1/(1-p) leaves the
    # mean unchanged. That rescaling is the whole trick; without it the merged
    # vector would shrink by a factor of (1-p).
    torch.manual_seed(0)
    taus = [{"w": torch.full((20000,), 2.0)}]
    out = dare(taus, drop_rate=0.9, base="task_arithmetic", scaling=1.0, seed=0)["w"]
    assert abs(out.mean().item() - 2.0) < 0.1, f"mean drifted to {out.mean().item():.3f}"


@check("every merger preserves keys and shapes")
def _():
    taus = [
        {"a": torch.randn(3, 4), "b": torch.randn(7)},
        {"a": torch.randn(3, 4), "b": torch.randn(7)},
    ]
    for name, fn in METHODS.items():
        out = fn(taus)
        assert sorted(out) == ["a", "b"], f"{name} returned keys {sorted(out)}"
        for k in ("a", "b"):
            assert out[k].shape == taus[0][k].shape, f"{name} changed the shape of {k}"


@check("mergers reject mismatched task vectors")
def _():
    good = {"a": torch.randn(4)}
    for bad, why in [({"b": torch.randn(4)}, "different keys"),
                     ({"a": torch.randn(5)}, "different shapes")]:
        try:
            weight_averaging([good, bad])
        except ValueError:
            continue
        raise AssertionError(f"accepted task vectors with {why}")


if __name__ == "__main__":
    print("self-test\n")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
