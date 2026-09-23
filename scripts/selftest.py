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

from mergeability.data import (
    MIXING_LEVELS,
    _random_crop,
    _semantic_task_classes,
    batches,
    build_tasks,
    load_cifar100_raw,
    semantic_purity,
    to_model_input,
)
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
from mergeability.similarity import (
    cosine_similarity,
    linear_cka,
    pairwise_summary,
    sign_conflict_rate,
    subspace_overlap,
    tau_norm,
)

PASSED, FAILED, SKIPPED = [], [], []

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
HAVE_DATA = (DATA_ROOT / "cifar-100-python").exists()


def needs_data(fn):
    """Mark a check that cannot run before CIFAR-100 has been downloaded."""
    fn.needs_data = True
    return fn


def check(name: str):
    def deco(fn):
        if getattr(fn, "needs_data", False) and not HAVE_DATA:
            SKIPPED.append(name)
            print(f"  skip  {name} (dataset not downloaded yet)")
            return fn
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


# --------------------------------------------------------------------------
# pre-merge signals
# --------------------------------------------------------------------------

@check("cosine of a task vector with itself is 1, with its negation -1")
def _():
    t = {"w": torch.randn(500)}
    neg = {"w": -t["w"]}
    assert abs(cosine_similarity(t, t) - 1.0) < 1e-5
    assert abs(cosine_similarity(t, neg) + 1.0) < 1e-5


@check("cosine is invariant to rescaling either task vector")
def _():
    # Rescaling changes ||tau|| but not direction; a signal that confused the
    # two would be baseline b2 wearing a disguise.
    a = {"w": torch.randn(500)}
    b = {"w": torch.randn(500)}
    plain = cosine_similarity(a, b)
    scaled = cosine_similarity({"w": a["w"] * 17.0}, {"w": b["w"] * 0.03})
    assert abs(plain - scaled) < 1e-5, f"{plain:.6f} vs {scaled:.6f}"


@check("sign conflict is 0 against itself and 1 against its negation")
def _():
    t = {"w": torch.randn(500)}
    neg = {"w": -t["w"]}
    assert sign_conflict_rate(t, t) < 1e-6
    assert abs(sign_conflict_rate(t, neg) - 1.0) < 1e-6


@check("unweighted sign conflict of independent vectors is about one half")
def _():
    torch.manual_seed(0)
    a, b = {"w": torch.randn(20000)}, {"w": torch.randn(20000)}
    r = sign_conflict_rate(a, b, weighted=False)
    assert 0.47 < r < 0.53, f"{r:.3f}"


@check("sign conflict weighting follows magnitude, not coordinate count")
def _():
    # One huge coordinate agrees, many tiny ones disagree. Counting coordinates
    # says "mostly conflict"; weighting by magnitude says the opposite, and for
    # merging it is the magnitudes that do the damage.
    a = {"w": torch.tensor([100.0] + [0.01] * 99)}
    b = {"w": torch.tensor([100.0] + [-0.01] * 99)}
    assert sign_conflict_rate(a, b, weighted=False) > 0.95
    assert sign_conflict_rate(a, b, weighted=True) < 0.05


@check("subspace overlap is 1 for identical updates, ~0 for orthogonal ones")
def _():
    torch.manual_seed(0)
    a = {"w": torch.randn(32, 32)}
    assert abs(subspace_overlap(a, a, k=4) - 1.0) < 1e-4
    # a rank-4 update and another supported on disjoint rows
    u = torch.zeros(32, 32); u[:4] = torch.randn(4, 32)
    v = torch.zeros(32, 32); v[4:8] = torch.randn(4, 32)
    assert subspace_overlap({"w": u}, {"w": v}, k=4) < 1e-4


@check("subspace overlap skips 1-D parameters")
def _():
    a = {"w": torch.randn(16, 16), "bias": torch.randn(16)}
    b = {"w": torch.randn(16, 16), "bias": torch.randn(16)}
    assert 0.0 <= subspace_overlap(a, b, k=4) <= 1.0  # would be nan if bias leaked


@check("CKA of a representation with itself is 1")
def _():
    x = torch.randn(200, 32)
    assert abs(linear_cka(x, x) - 1.0) < 1e-8


@check("CKA is invariant to rotation and isotropic scaling")
def _():
    # This invariance is the reason CKA works across independently trained
    # models at all: two encoders that agree up to a rotation of their latent
    # space are computing the same thing.
    torch.manual_seed(0)
    x = torch.randn(200, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert abs(linear_cka(x, x @ q) - 1.0) < 1e-6, "not rotation invariant"
    assert abs(linear_cka(x, x * 9.0) - 1.0) < 1e-6, "not scale invariant"


@check("CKA is symmetric and low for independent representations")
def _():
    torch.manual_seed(0)
    x, y = torch.randn(400, 32), torch.randn(400, 32)
    assert abs(linear_cka(x, y) - linear_cka(y, x)) < 1e-8, "not symmetric"
    assert linear_cka(x, y) < 0.3, f"independent features scored {linear_cka(x, y):.3f}"


@check("CKA compares representations of different width")
def _():
    torch.manual_seed(0)
    x = torch.randn(200, 32)
    assert 0.0 <= linear_cka(x, torch.randn(200, 64)) <= 1.0


@check("pairwise summary averages over all unordered pairs")
def _():
    taus = [{"w": torch.randn(64, 8)} for _ in range(4)]
    out = pairwise_summary(taus)
    assert out["num_tasks"] == 4.0
    for key in ("cosine", "sign_conflict", "subspace_overlap", "mean_tau_norm"):
        assert key in out and out[key] == out[key], f"{key} missing or nan"


@check("tau_norm matches a direct norm computation")
def _():
    t = {"a": torch.randn(10), "b": torch.randn(3, 4)}
    direct = torch.cat([t["a"], t["b"].reshape(-1)]).norm().item()
    assert abs(tau_norm(t) - direct) < 1e-5


# --------------------------------------------------------------------------
# the graded similarity axis (needs CIFAR-100 on disk)
# --------------------------------------------------------------------------

def _all_regimes():
    return {r: build_tasks(r, root=str(DATA_ROOT), download=False) for r in MIXING_LEVELS}


@check("every regime partitions the 100 classes into 5 tasks of 20")
@needs_data
def _():
    for regime, tasks in _all_regimes().items():
        assert len(tasks) == 5, f"{regime}: {len(tasks)} tasks"
        seen = sorted(c for t in tasks for c in t.fine_classes)
        assert seen == list(range(100)), f"{regime}: coverage is broken"
        for t in tasks:
            assert t.num_classes == 20, f"{regime}/{t.name}: {t.num_classes} classes"


@check("every regime gives every task the same training budget")
@needs_data
def _():
    # Task size must never vary with the mixing fraction, or the axis would be
    # confounded with how much data each task gets.
    for regime, tasks in _all_regimes().items():
        for t in tasks:
            assert len(t.y_train) == 10000, f"{regime}/{t.name}: {len(t.y_train)} train"
            assert len(t.y_test) == 2000, f"{regime}/{t.name}: {len(t.y_test)} test"


@check("alpha=0 reproduces the semantic partition exactly")
@needs_data
def _():
    raw = load_cifar100_raw(str(DATA_ROOT), download=False)
    semantic = _semantic_task_classes(raw)
    built = {t.name: t.fine_classes for t in build_tasks("semantic", root=str(DATA_ROOT), download=False)}
    assert sorted(semantic.values()) == sorted(built.values()), "semantic partition drifted"


@check("semantic purity falls monotonically as the mixing fraction rises")
@needs_data
def _():
    # The check that the axis actually interpolates rather than just being
    # differently named. Without this, "graded" would be an unverified claim.
    raw = load_cifar100_raw(str(DATA_ROOT), download=False)
    purity = {
        r: semantic_purity({t.name: t.fine_classes for t in tasks}, raw)
        for r, tasks in _all_regimes().items()
    }
    ordered = [purity[r] for r in sorted(MIXING_LEVELS, key=MIXING_LEVELS.get)]
    assert ordered[0] == 1.0, f"alpha=0 purity is {ordered[0]}, expected 1.0"
    for a, b in zip(ordered, ordered[1:]):
        assert b <= a + 1e-9, f"purity went up: {ordered}"
    assert ordered[0] - ordered[-1] > 0.5, f"axis barely moves: {ordered}"


@check("an unknown regime is rejected rather than silently guessed")
@needs_data
def _():
    try:
        build_tasks("mix33", root=str(DATA_ROOT), download=False)
    except ValueError:
        return
    raise AssertionError("accepted an undefined regime")


if __name__ == "__main__":
    print("self-test\n")
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed{tail}")
    sys.exit(1 if FAILED else 0)
