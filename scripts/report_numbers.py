#!/usr/bin/env python
"""Every number quoted in the report, as LaTeX macros -- none typed by hand.

Reads the raw results and the tables written by ``run_analysis.py`` and writes

* ``report/numbers.tex`` -- one ``\\newcommand`` per quoted number

    uv run scripts/run_analysis.py && uv run scripts/report_numbers.py

Correlations are printed with a proper minus sign (``\\ensuremath``), so the
macros work in running text and in math alike.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mergeability.analysis import ALGORITHMS, PREREGISTERED, outcomes
from mergeability.data import (
    _mixed_task_classes,
    _random_task_classes,
    _semantic_task_classes,
    load_cifar100_raw,
    semantic_purity,
)

RAW, ANA, REPORT = ROOT / "results/raw", ROOT / "results/analysis", ROOT / "report"
FAMILY = ["cosine", "sign_conflict", "subspace_overlap", "cka"]   # the collinear four on ViT
LEVELS = ["semantic", "mix25", "mix50", "mix75", "random"]

macros: dict[str, str] = {}


def put(name: str, value: str) -> None:
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"LaTeX macro names are letters only: {name!r}")
    if name in macros:
        raise ValueError(f"macro {name!r} defined twice")
    macros[name] = value


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def num(x: float, digits: int = 2) -> str:
    s = f"{x:.{digits}f}"
    return f"\\ensuremath{{-{s[1:]}}}" if s.startswith("-") else s


def count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def span(prefix: str, values) -> None:
    v = np.asarray(list(values), dtype=float)
    put(prefix + "Min", num(v.min()))
    put(prefix + "Max", num(v.max()))


def main() -> None:
    vit_runs = pd.read_csv(RAW / "merge_vit_full.csv")
    res_runs = pd.read_csv(RAW / "merge_resnet.csv")
    ft = pd.read_csv(RAW / "finetune_from_logs.csv")

    # ---- scale of the study ---------------------------------------------------
    n_vit = int((ft.backbone == "vit_tiny_patch16_224").sum())
    n_res = int((ft.backbone == "resnet18").sum())
    put("nFTvit", str(n_vit))
    put("nFTres", str(n_res))
    put("nFTall", str(n_vit + n_res))
    put("nMergeVit", count(len(vit_runs)))
    put("nMergeRes", count(len(res_runs)))
    put("nMergeAll", count(len(vit_runs) + len(res_runs)))
    put("nConfigs", str(int(vit_runs.groupby(["regime", "seed", "subset"]).size().max())))

    # ---- resolution check (mean linear-probe accuracy over the probed tasks) --
    def probe_at(fname: str, res: int) -> float:
        rows = json.loads((RAW / fname).read_text())["rows"]
        return float(np.mean([r["probe_acc"] for r in rows if r["resolution"] == res]))
    put("resVitLow", pct(probe_at("resolution_vit_fixed.json", 128)))
    put("resVitHigh", pct(probe_at("resolution_vit_fixed.json", 224)))
    put("resResLow", pct(probe_at("resolution_resnet.json", 128)))
    put("resResHigh", pct(probe_at("resolution_resnet.json", 224)))

    # ---- the similarity dial: semantic purity per level -------------------------
    raw = load_cifar100_raw(ROOT / "data", download=False)
    names = {"semantic": "Semantic", "mix25": "MixA", "mix50": "MixB",
             "mix75": "MixC", "random": "Random"}
    for level, alpha in zip(LEVELS, [0.0, 0.25, 0.5, 0.75, 1.0]):
        if level == "semantic":
            classes = _semantic_task_classes(raw)
        elif level == "random":
            classes = _random_task_classes(5, 0)
        else:
            classes = _mixed_task_classes(raw, alpha, 0, prefix=level)
        put("purity" + names[level], f"{100 * semantic_purity(classes, raw):.0f}")

    # ---- fine-tuning ---------------------------------------------------------------
    for tag, b in [("Vit", "vit_tiny_patch16_224"), ("Res", "resnet18")]:
        f = ft[ft.backbone == b]
        gain = f.finetuned_acc - f.probe_acc
        put(f"ft{tag}Probe", f"{f.probe_acc.mean():.1f}")
        put(f"ft{tag}Final", f"{f.finetuned_acc.mean():.1f}")
        put(f"ft{tag}GainMin", f"{gain.min():.1f}")
        put(f"ft{tag}GainMax", f"{gain.max():.1f}")

    # ---- merge outcomes ------------------------------------------------------------
    o = outcomes(vit_runs)
    o["mean_alg"] = o[ALGORITHMS].mean(axis=1)
    for level, tag in [("semantic", "Sem"), ("random", "Rand")]:
        for n, ntag in [(2, "Two"), (5, "Five")]:
            v = o[(o.regime == level) & (o.n_tasks == n)].mean_alg
            put(f"dial{tag}{ntag}", pct(v.mean()))

    # the floor of merging at its best setting (ResNet's DARE has lambda = 1 only)
    put("tunedMinVit", pct(o[ALGORITHMS].min().min()))
    res_tuned = outcomes(res_runs)
    put("tunedMinRes", pct(res_tuned[["averaging", "task_arithmetic", "ties"]].min().min()))

    five = vit_runs[vit_runs.n_tasks == 5]
    by_scale = five.groupby(["method", "param_scaling"]).normalized_acc.mean()
    put("taOneFive", pct(by_scale["task_arithmetic", 1.0]))
    put("dareOneFive", pct(by_scale["dare", 1.0]))
    dare_best_s = by_scale["dare"].idxmax()
    put("dareBestScale", f"{dare_best_s:g}")
    put("dareBestFive", pct(by_scale["dare", dare_best_s]))
    put("taAtDareBestFive", pct(by_scale["task_arithmetic", dare_best_s]))
    swept = [s for s in by_scale["dare"].index if s < 1.0]
    gap = max(abs(by_scale["dare", s] - by_scale["task_arithmetic", s]) for s in swept)
    put("dareTaGapMax", f"{100 * gap:.1f}")

    # ---- signals (tuned outcome) ---------------------------------------------------
    sig = pd.read_csv(ANA / "signals.csv")

    def cells(backbone: str, signal: str | list[str], col: str = "rho", absolute=False):
        signals = [signal] if isinstance(signal, str) else signal
        v = sig[(sig.backbone == backbone) & sig.signal.isin(signals)][col]
        return v.abs() if absolute else v

    span("rhoVitFam", cells("vit", FAMILY, absolute=True))
    span("partVitFam", cells("vit", FAMILY, "partial_rho_given_norm", absolute=True))
    put("rhoVitDriftAbsMax", num(cells("vit", "drift", absolute=True).max()))
    span("rhoVitNorm", cells("vit", "mean_tau_norm"))
    span("rhoVitCka", cells("vit", "cka"))
    span("partVitCka", cells("vit", "cka", "partial_rho_given_norm"))
    ci = sig[(sig.backbone == "vit") & (sig.signal == "cka") & (sig.algorithm == "task_arithmetic")].iloc[0]
    put("ciVitCkaLo", num(ci.ci_lo))
    put("ciVitCkaHi", num(ci.ci_hi))

    span("rhoResCos", cells("resnet", "cosine"))
    span("partResCos", cells("resnet", "cosine", "partial_rho_given_norm"))
    span("rhoResCka", cells("resnet", "cka"))
    span("partResCka", cells("resnet", "cka", "partial_rho_given_norm"))
    span("rhoResDrift", cells("resnet", "drift"))
    span("rhoResNorm", cells("resnet", "mean_tau_norm"))
    span("rhoResSub", cells("resnet", "subspace_overlap"))

    within = pd.read_csv(ANA / "signals_within_regime_vit.csv", index_col=0)
    span("rhoVitWithin", within.loc[FAMILY].abs().to_numpy().ravel())

    fixed = pd.read_csv(ANA / "signals_fixed.csv")
    delta = (sig.set_index(["backbone", "algorithm", "signal"]).rho
             - fixed.set_index(["backbone", "algorithm", "signal"]).rho).abs()
    put("fixedMaxDelta", num(delta.max()))

    # ---- redundancy ------------------------------------------------------------------
    cv = pd.read_csv(ANA / "signal_correlations_vit.csv", index_col=0)
    cr = pd.read_csv(ANA / "signal_correlations_resnet.csv", index_col=0)
    put("corrCosSignVit", num(cv.loc["cosine", "sign_conflict"]))
    put("corrCosSignRes", num(cr.loc["cosine", "sign_conflict"]))
    fam = cv.loc[FAMILY, FAMILY].abs().to_numpy()
    put("corrVitFamMin", num(fam[~np.eye(len(FAMILY), dtype=bool)].min()))

    # ---- stability (§6) --------------------------------------------------------------
    st = pd.read_csv(ANA / "stability.csv", index_col=0)
    span("kendallSeedVit", st.seeds_vit)
    span("kendallSeedRes", st.seeds_resnet)
    ag = pd.read_csv(ANA / "agreement_algorithms_resnet.csv", index_col=0).to_numpy()
    put("kendallAlgResMin", num(ag[~np.eye(len(ag), dtype=bool)].min()))

    # ---- tables ----------------------------------------------------------------------
    ridge = pd.read_csv(ANA / "ridge.csv")
    r = ridge[(ridge.backbone == "vit") & (ridge.held_out == "regime")]
    span("rTwoBTwo", r[r.model == "b2: + norm"].r2)
    span("rTwoCka", r[r.model == "+ cka"].r2)
    # rho of each algorithm's pre-registered signal on ViT (DARE's was not computed)
    macro_name = {"averaging": "Averaging", "task_arithmetic": "TaskArithmetic", "ties": "Ties"}
    for alg, tag in macro_name.items():
        row = sig[(sig.backbone == "vit") & (sig.algorithm == alg) & (sig.signal == PREREGISTERED[alg])]
        put("rhoPre" + tag, num(row.rho.iloc[0]))

    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items())
    (REPORT / "numbers.tex").write_text(
        "% Generated by scripts/report_numbers.py. Do not edit.\n" + body + "\n")
    print(f"wrote {len(macros)} macros to report/numbers.tex")
    for k, v in macros.items():
        print(f"  \\{k:20s} {v}")


if __name__ == "__main__":
    main()
