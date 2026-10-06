r"""From merge results to the answer: which pre-merge signal predicts the damage,
for which algorithm, and how stable that is.

This implements PROJECT_PLAN §8 (items 1-3) and the stability control of §6 on
the CSVs written by ``run_merge.py``. Three choices shape every number, so they
are made here once and stated:

**One outcome per (subset, algorithm).** An algorithm with hyperparameters is
scored at its best configuration for that subset ("tuned"), which is how task
arithmetic's lambda is used in practice. DARE's seeds are replicates of one
configuration, not configurations, so they are averaged *before* taking the
best -- otherwise DARE would be credited with its luckiest draw. ``mode="fixed"``
instead uses, per number of tasks, the single configuration that is best on
average: the robustness check that per-subset tuning is not doing the work.

**Correlations within each number of tasks.** Merging more tasks is worse, and
several signals move with the number of tasks too, so a pooled correlation
would partly measure "more tasks is worse" -- which is baseline ``b1``. Every
correlation is therefore computed inside one subset size and averaged across
sizes (weighted by count). ``b1`` has no variance inside a stratum and scores
zero by construction; that is the point. Strata with too few subsets (the
single 5-task merge per group) are left out.

**Beyond distance travelled.** Baseline ``b2`` is the mean task-vector norm. A
signal that only restates how far fine-tuning moved the weights will lose its
correlation once ``b2`` is partialled out, so each signal is also reported as a
partial rank correlation controlling for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PREDICTORS = ["cosine", "sign_conflict", "subspace_overlap", "cka", "drift"]
NORM = "mean_tau_norm"                       # baseline b2
ALGORITHMS = ["averaging", "task_arithmetic", "ties", "dare"]
KEYS = ["backbone", "regime", "seed", "subset", "n_tasks"]

# PROJECT_PLAN §4: the signal each algorithm's mechanism predicts will matter.
# DARE's (task-vector density / effective rank) is an appendix signal and is
# not among the five main predictors.
PREREGISTERED = {
    "averaging": "drift",
    "task_arithmetic": "cosine",
    "ties": "sign_conflict",
    "dare": None,
}

_CONFIG = ["param_scaling", "param_density", "param_drop_rate"]


# --------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------

def outcomes(runs: pd.DataFrame, mode: str = "tuned",
             dare_scaling: float | None = None) -> pd.DataFrame:
    """One row per merged subset: its signals and one outcome per algorithm.

    ``dare_scaling`` restricts DARE to a single scaling, for like-for-like
    comparisons with a run that only has that one (the ResNet-18 control).
    """
    if mode not in ("tuned", "fixed"):
        raise ValueError(f"unknown mode {mode!r}")
    runs = runs.copy()
    if dare_scaling is not None:
        runs = runs[(runs.method != "dare") | np.isclose(runs.param_scaling, dare_scaling)]
    runs[_CONFIG] = runs[_CONFIG].fillna(-1.0)          # groupby drops NaN keys

    # DARE seeds -> one number per configuration (param_seed is not a key)
    per_cfg = runs.groupby(KEYS + ["method"] + _CONFIG, as_index=False).normalized_acc.mean()

    if mode == "tuned":
        best = per_cfg.groupby(KEYS + ["method"]).normalized_acc.max()
    else:
        by = ["backbone", "n_tasks", "method"]
        avg = per_cfg.groupby(by + _CONFIG).normalized_acc.mean().reset_index()
        chosen = avg.loc[avg.groupby(by).normalized_acc.idxmax(), by + _CONFIG]
        best = per_cfg.merge(chosen, on=by + _CONFIG).set_index(KEYS + ["method"]).normalized_acc

    wide = best.unstack("method")
    missing = [a for a in ALGORITHMS if a not in wide]
    if missing:
        raise ValueError(f"runs lack algorithms {missing}")
    signals = runs.drop_duplicates(KEYS).set_index(KEYS)[PREDICTORS + [NORM]]
    return wide[ALGORITHMS].join(signals).reset_index()


# --------------------------------------------------------------------------
# rank statistics inside strata
# --------------------------------------------------------------------------

def _strata(n: np.ndarray, min_size: int):
    for s in np.unique(n):
        m = n == s
        if m.sum() >= min_size:
            yield m


def stratified_spearman(x, y, strata, min_size: int = 20) -> float:
    """Spearman rho inside each stratum, averaged weighted by stratum size."""
    x, y, strata = map(np.asarray, (x, y, strata))
    rhos, w = [], []
    for m in _strata(strata, min_size):
        r = spearmanr(x[m], y[m]).statistic
        if np.isfinite(r):
            rhos.append(r)
            w.append(m.sum())
    return float(np.average(rhos, weights=w)) if rhos else float("nan")


def _residual(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones_like(z), z])
    return a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]


def partial_spearman(x, y, z, strata, min_size: int = 20) -> float:
    """Rank correlation of x and y with z partialled out, inside each stratum.

    Ranks are taken within the stratum, then x and y are each residualised on
    z's ranks; the correlation of the residuals is what x says about y that z
    does not already say.
    """
    x, y, z, strata = map(np.asarray, (x, y, z, strata))
    rhos, w = [], []
    for m in _strata(strata, min_size):
        rx, ry, rz = (pd.Series(v[m]).rank().to_numpy() for v in (x, y, z))
        ex, ey = _residual(rx, rz), _residual(ry, rz)
        denom = np.sqrt((ex @ ex) * (ey @ ey))
        if denom > 0:
            rhos.append(float(ex @ ey / denom))
            w.append(m.sum())
    return float(np.average(rhos, weights=w)) if rhos else float("nan")


def bootstrap_ci(stat, arrays: list[np.ndarray], strata, n_boot: int = 2000,
                 seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, resampling subsets with replacement inside strata.

    Subsets that share tasks are not independent, so this interval is on the
    optimistic side; the seed-split stability check is the stricter test.
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g)) for g in groups])
        vals.append(stat(*[np.asarray(a)[idx] for a in arrays], strata[idx]))
    lo, hi = np.nanpercentile(vals, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# §8 item 1: every signal against every algorithm
# --------------------------------------------------------------------------

def signal_table(o: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Stratified Spearman (with CI) and partial-on-b2 rho, per algorithm x signal."""
    rows = []
    n = o.n_tasks.to_numpy()
    for alg in ALGORITHMS:
        y = o[alg].to_numpy()
        for sig in PREDICTORS + [NORM]:
            x = o[sig].to_numpy()
            rho = stratified_spearman(x, y, n)
            lo, hi = bootstrap_ci(stratified_spearman, [x, y], n, n_boot=n_boot)
            partial = (partial_spearman(x, y, o[NORM].to_numpy(), n)
                       if sig != NORM else float("nan"))
            rows.append({"algorithm": alg, "signal": sig, "rho": rho,
                         "ci_lo": lo, "ci_hi": hi, "partial_rho_given_norm": partial,
                         "preregistered": PREREGISTERED[alg] == sig})
    return pd.DataFrame(rows)


def signal_correlations(o: pd.DataFrame) -> pd.DataFrame:
    """Stratified Spearman between the signals themselves: which are redundant."""
    sigs = PREDICTORS + [NORM]
    n = o.n_tasks.to_numpy()
    return pd.DataFrame(
        [[stratified_spearman(o[a], o[b], n) for b in sigs] for a in sigs],
        index=sigs, columns=sigs,
    )


# --------------------------------------------------------------------------
# §8 item 3 + §6: rankings and their stability
# --------------------------------------------------------------------------

def importance(o: pd.DataFrame) -> pd.DataFrame:
    """|stratified rho| of each predictor, per algorithm (predictors x algorithms)."""
    n = o.n_tasks.to_numpy()
    return pd.DataFrame(
        {alg: [abs(stratified_spearman(o[p], o[alg], n)) for p in PREDICTORS]
         for alg in ALGORITHMS},
        index=PREDICTORS,
    )


def ranking_agreement(a: pd.Series, b: pd.Series) -> float:
    """Kendall tau between two predictor-importance vectors (same index)."""
    return float(kendalltau(a.loc[PREDICTORS], b.loc[PREDICTORS]).statistic)


def stability(o: pd.DataFrame, other: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """The §6 test: rankings should agree across seeds (and backbones), and
    only then may their disagreement across algorithms be read as a finding.

    Returns Kendall tau between importance rankings: seed 0 vs seed 1 per
    algorithm; ``o`` vs ``other`` per algorithm (if given); and every pair of
    algorithms within ``o``.
    """
    imp = importance(o)
    by_seed = {s: importance(o[o.seed == s]) for s in sorted(o.seed.unique())}
    s0, s1 = list(by_seed.values())[:2]
    out = {
        "across_seeds": pd.DataFrame(
            {"kendall_tau": [ranking_agreement(s0[a], s1[a]) for a in ALGORITHMS]},
            index=ALGORITHMS),
        "across_algorithms": pd.DataFrame(
            [[ranking_agreement(imp[a], imp[b]) for b in ALGORITHMS] for a in ALGORITHMS],
            index=ALGORITHMS, columns=ALGORITHMS),
    }
    if other is not None:
        imp2 = importance(other)
        out["across_backbones"] = pd.DataFrame(
            {"kendall_tau": [ranking_agreement(imp[a], imp2[a]) for a in ALGORITHMS]},
            index=ALGORITHMS)
    return out


# --------------------------------------------------------------------------
# §8 item 2: cross-validated ridge against the baselines
# --------------------------------------------------------------------------

def ridge_cv(o: pd.DataFrame, algorithm: str, features: list[str], group: str,
             alphas=(0.01, 0.1, 1.0, 10.0, 100.0)) -> dict[str, float]:
    """Out-of-group R^2 of ridge on ``n_tasks`` (one-hot) + ``features``.

    ``group`` is the column held out in turn: ``regime`` asks whether the
    signals carry the similarity axis to a level never seen in training;
    ``seed`` whether they carry it to an unseen fine-tuning run.
    ``features=[]`` is baseline b1; ``[NORM]`` is b2 (on top of b1).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    y = o[algorithm].to_numpy()
    onehot = pd.get_dummies(o.n_tasks, prefix="n").to_numpy(dtype=float)
    pred = np.empty_like(y)
    for g in o[group].unique():
        test = (o[group] == g).to_numpy()
        X_parts_tr, X_parts_te = [onehot[~test]], [onehot[test]]
        if features:
            scaler = StandardScaler().fit(o.loc[~test, features])
            X_parts_tr.append(scaler.transform(o.loc[~test, features]))
            X_parts_te.append(scaler.transform(o.loc[test, features]))
        model = RidgeCV(alphas=alphas).fit(np.hstack(X_parts_tr), y[~test])
        pred[test] = model.predict(np.hstack(X_parts_te))
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"r2": float(1 - ss_res / ss_tot),
            "rho_within_n": stratified_spearman(pred, y, o.n_tasks.to_numpy())}
