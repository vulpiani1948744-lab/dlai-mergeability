#!/usr/bin/env python
"""Which pre-merge signal predicts merging damage, per algorithm -- and is that stable?

Reads the merge CSVs, prints the tables of PROJECT_PLAN §8 (items 1-3) and the
stability control of §6, and writes every table to ``results/analysis/`` so the
report can be generated from files rather than typed by hand.

    uv run scripts/run_analysis.py
    uv run scripts/run_analysis.py --mode fixed      # robustness: no per-subset tuning

The ResNet-18 control only has DARE at scaling 1.0, so every ViT-vs-ResNet
comparison restricts ViT to the same regimes and the same DARE scaling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergeability.analysis import (
    ALGORITHMS,
    NORM,
    PREDICTORS,
    importance,
    outcomes,
    ridge_cv,
    signal_correlations,
    signal_table,
    stability,
    stratified_spearman,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vit", type=Path, default=Path("results/raw/merge_vit_full.csv"))
    p.add_argument("--resnet", type=Path, default=Path("results/raw/merge_resnet.csv"))
    p.add_argument("--mode", choices=["tuned", "fixed"], default="tuned")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("results/analysis"))
    return p.parse_args()


def show(title: str, df: pd.DataFrame, digits: int = 3) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(df.round(digits).to_string())


def signal_view(t: pd.DataFrame, col: str) -> pd.DataFrame:
    v = t.pivot(index="signal", columns="algorithm", values=col)
    return v.loc[PREDICTORS + [NORM], ALGORITHMS]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.mode == "tuned" else f"_{args.mode}"
    pd.set_option("display.width", 140)

    vit = outcomes(pd.read_csv(args.vit), mode=args.mode)
    res = outcomes(pd.read_csv(args.resnet), mode=args.mode)
    # like-for-like ViT for the backbone comparison: ResNet's regimes, DARE at 1.0
    vit_like = outcomes(pd.read_csv(args.vit), mode=args.mode, dare_scaling=1.0)
    vit_like = vit_like[vit_like.regime.isin(res.regime.unique())]

    print(f"outcome per (subset, algorithm): {args.mode}")
    for name, o in [("ViT-Tiny", vit), ("ResNet-18", res)]:
        print(f"  {name}: {len(o)} subsets, "
              f"{o.n_tasks.value_counts().sort_index().to_dict()} by number of tasks, "
              f"regimes {sorted(o.regime.unique())}")
    print("correlations use pairs and triples; each 5-task stratum is too small")

    # ---- §8.1 every signal against every algorithm --------------------------
    tables = {}
    for name, o in [("vit", vit), ("resnet", res)]:
        t = signal_table(o, n_boot=args.n_boot)
        t.insert(0, "backbone", name)
        tables[name] = t
        show(f"[{name}] Spearman rho within number of tasks (signal x algorithm)",
             signal_view(t, "rho"))
        ci = t.assign(ci=t.apply(lambda r: f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", axis=1))
        show(f"[{name}] 95% bootstrap interval", signal_view(ci, "ci"))
        show(f"[{name}] partial rho, controlling for {NORM} (baseline b2)",
             signal_view(t, "partial_rho_given_norm"))
    pd.concat(tables.values()).to_csv(args.out / f"signals{suffix}.csv", index=False)

    # The regimes are the designed similarity dial, so most of the variance above
    # is between regimes. Within one regime and one subset size, what is left is
    # the variation between subsets of the same task family.
    for name, o in [("vit", vit), ("resnet", res)]:
        strata = (o.regime + "/" + o.n_tasks.astype(str)).to_numpy()
        w = pd.DataFrame(
            {a: [stratified_spearman(o[s], o[a], strata) for s in PREDICTORS + [NORM]]
             for a in ALGORITHMS},
            index=PREDICTORS + [NORM])
        show(f"[{name}] Spearman rho within regime and number of tasks", w)
        w.to_csv(args.out / f"signals_within_regime_{name}{suffix}.csv")

    for name, o in [("vit", vit), ("resnet", res)]:
        c = signal_correlations(o)
        show(f"[{name}] signals against each other (redundancy)", c, 2)
        c.to_csv(args.out / f"signal_correlations_{name}{suffix}.csv")

    # ---- §8.3 + §6 rankings and their stability ----------------------------
    for name, o in [("vit", vit), ("resnet", res)]:
        imp = importance(o)
        ranks = imp.rank(ascending=False).astype(int)
        show(f"[{name}] importance rank (1 = strongest |rho|)", ranks, 0)
        imp.to_csv(args.out / f"importance_{name}{suffix}.csv")

    stab_vit = stability(vit)
    stab_like = stability(vit_like, other=res)
    stab_res = stability(res)
    rows = []
    for alg in ALGORITHMS:
        rows.append({
            "algorithm": alg,
            "seeds_vit": stab_vit["across_seeds"].loc[alg, "kendall_tau"],
            "seeds_resnet": stab_res["across_seeds"].loc[alg, "kendall_tau"],
            "vit_vs_resnet": stab_like["across_backbones"].loc[alg, "kendall_tau"],
        })
    st = pd.DataFrame(rows).set_index("algorithm")
    show("§6 stability: Kendall tau between importance rankings (1 = identical)", st, 2)
    show("[vit] agreement between algorithms' rankings", stab_vit["across_algorithms"], 2)
    show("[resnet] agreement between algorithms' rankings", stab_res["across_algorithms"], 2)
    st.to_csv(args.out / f"stability{suffix}.csv")
    stab_vit["across_algorithms"].to_csv(args.out / f"agreement_algorithms_vit{suffix}.csv")
    stab_res["across_algorithms"].to_csv(args.out / f"agreement_algorithms_resnet{suffix}.csv")

    # ---- §8.2 cross-validated ridge against the baselines -------------------
    models = {"b1: n_tasks": [], "b2: + norm": [NORM]}
    models.update({f"+ {p}": [p] for p in PREDICTORS})
    models["+ all five"] = PREDICTORS
    models["+ all five + norm"] = PREDICTORS + [NORM]
    rows = []
    for name, o, groups in [("vit", vit, ["regime", "seed"]), ("resnet", res, ["seed"])]:
        for group in groups:
            for alg in ALGORITHMS:
                for label, feats in models.items():
                    r = ridge_cv(o, alg, feats, group)
                    rows.append({"backbone": name, "held_out": group, "algorithm": alg,
                                 "model": label, **r})
    ridge = pd.DataFrame(rows)
    ridge.to_csv(args.out / f"ridge{suffix}.csv", index=False)
    for (name, group), g in ridge.groupby(["backbone", "held_out"], sort=False):
        v = g.pivot(index="model", columns="algorithm", values="r2").loc[list(models), ALGORITHMS]
        show(f"[{name}] ridge, out-of-{group} R^2", v)

    print(f"\nwrote tables to {args.out}/")


if __name__ == "__main__":
    main()
