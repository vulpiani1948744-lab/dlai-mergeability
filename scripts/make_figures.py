#!/usr/bin/env python
"""The report's two figures, drawn from the result files.

    uv run scripts/run_analysis.py && uv run scripts/make_figures.py

* ``report/fig_dial.pdf``    -- the similarity dial, and DARE vs task arithmetic
* ``report/fig_signals.pdf`` -- Spearman rho, signal x algorithm, per backbone

Colours: an ordinal one-hue ramp for subset size, neutral ink vs one accent for
the two-series panel, and a blue<->red diverging scale with a grey midpoint for
correlations (poles matched in OKLab lightness so neither sign looks stronger).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mergeability.analysis import ALGORITHMS, NORM, PREDICTORS, PREREGISTERED, outcomes

OUT = ROOT / "report"
PREVIEW = ROOT / "results" / "analysis"
COL_W, TEXT_W = 3.25, 6.75          # inches, two-column layout
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SIZE_RAMP = {2: "#86b6ef", 3: "#2a78d6", 5: "#104281"}   # validated ordinal ramp
ACCENT = "#eb6834"
LEVELS = ["semantic", "mix25", "mix50", "mix75", "random"]
ALPHA = [0.0, 0.25, 0.5, 0.75, 1.0]
ALG_LABEL = {"averaging": "Avg.", "task_arithmetic": "TA", "ties": "TIES", "dare": "DARE"}
SIG_LABEL = {"cosine": "cosine", "sign_conflict": "sign conflict",
             "subspace_overlap": "subspace overlap", "cka": "CKA",
             "drift": "drift (CKA to $\\theta_0$)", NORM: r"$\Vert\tau\Vert$ (baseline $b_2$)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


# --------------------------------------------------------------------------
# OKLab, for a lightness-matched diverging scale
# --------------------------------------------------------------------------

def _to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_oklab(h: str) -> np.ndarray:
    r, g, b = _to_lin([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = np.cbrt([l, m, s])
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    return rgb


def in_gamut(lab) -> bool:
    rgb = oklab_to_rgb(lab)
    return bool(np.all(rgb >= -1e-4) and np.all(rgb <= 1 + 1e-4))


def matched(step_hex: str, hue_hex: str) -> np.ndarray:
    """The colour with ``step_hex``'s lightness and chroma, in ``hue_hex``'s hue."""
    L, a, b = hex_to_oklab(step_hex)
    chroma = np.hypot(a, b)
    _, ha, hb = hex_to_oklab(hue_hex)
    hue = np.arctan2(hb, ha)
    while chroma > 0 and not in_gamut([L, chroma * np.cos(hue), chroma * np.sin(hue)]):
        chroma *= 0.97
    return np.array([L, chroma * np.cos(hue), chroma * np.sin(hue)])


def diverging_cmap() -> ListedColormap:
    blue = ["#104281", "#2a78d6", "#86b6ef", "#cde2fb"]            # dark -> light
    red = [matched(h, "#e34948") for h in blue]
    mid = hex_to_oklab("#f0efec")
    nodes = [*red, mid, *[hex_to_oklab(h) for h in reversed(blue)]]  # -1 ... +1
    xs = np.linspace(0, 1, len(nodes))
    grid = np.linspace(0, 1, 256)
    labs = np.array([[np.interp(g, xs, [n[k] for n in nodes]) for k in range(3)] for g in grid])
    return ListedColormap([_to_srgb(oklab_to_rgb(lab)) for lab in labs])


# --------------------------------------------------------------------------
# figure 1
# --------------------------------------------------------------------------

def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_dial(vit_runs: pd.DataFrame) -> None:
    o = outcomes(vit_runs)
    o["y"] = o[ALGORITHMS].mean(axis=1) * 100
    per_seed = o.groupby(["n_tasks", "regime", "seed"]).y.mean().reset_index()

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.55),
                               gridspec_kw={"wspace": 0.42})
    for n, color in SIZE_RAMP.items():
        g = per_seed[per_seed.n_tasks == n].groupby("regime").y
        mean, lo, hi = (g.agg(f).loc[LEVELS].to_numpy() for f in ("mean", "min", "max"))
        a.errorbar(ALPHA, mean, yerr=[mean - lo, hi - mean], color=color, lw=1.3,
                   marker="o", ms=3.5, mec="white", mew=0.6, capsize=1.5,
                   elinewidth=0.7, label=f"{n} tasks")
        a.annotate(str(n), (ALPHA[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=6.5, color=INK_2)
    a.set_xticks(ALPHA, ["0\nsem.", ".25", ".5", ".75", "1\nrand."])
    a.set_xlim(-0.08, 1.12)
    a.set_xlabel("mixing fraction $\\alpha$")
    a.set_ylabel("normalized accuracy (%)")
    a.set_title("(a) the similarity dial", loc="left")
    a.legend(frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.2)
    style(a)

    five = vit_runs[vit_runs.n_tasks == 5]
    for method, color, label in [("task_arithmetic", INK_2, "task arithmetic"),
                                 ("dare", ACCENT, "DARE")]:
        s = five[five.method == method].groupby("param_scaling").normalized_acc.mean() * 100
        b.plot(s.index, s.values, color=color, lw=1.3, marker="o", ms=3.5,
               mec="white", mew=0.6, label=label)
    b.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0], [".1", ".2", ".3", ".5", "1"])
    b.set_xlabel("scaling $\\lambda$")
    b.set_ylabel("normalized accuracy (%)")
    b.set_title("(b) 5 tasks: DARE vs TA", loc="left")
    b.legend(frameon=False, loc="center left", bbox_to_anchor=(0, 0.42), handlelength=1.2, borderaxespad=0.2)
    style(b)

    fig.savefig(OUT / "fig_dial.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "fig_dial.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 2
# --------------------------------------------------------------------------

def fig_signals(sig: pd.DataFrame) -> None:
    cmap = diverging_cmap()
    rows = PREDICTORS + [NORM]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 1.6), gridspec_kw={"wspace": 0.05})
    for k, (ax, backbone, title) in enumerate(zip(
            axes, ["vit", "resnet"], ["ViT-Tiny, 5 similarity levels",
                                      "ResNet-18, semantic + random"])):
        m = (sig[sig.backbone == backbone]
             .pivot(index="signal", columns="algorithm", values="rho")
             .loc[rows, ALGORITHMS])
        im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i, s in enumerate(rows):
            for j, alg in enumerate(ALGORITHMS):
                v = m.loc[s, alg]
                lab = hex_to_oklab(matplotlib.colors.to_hex(cmap((v + 1) / 2)))
                ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center",
                        fontsize=6.5, color=INK if lab[0] > 0.62 else "white")
                if PREREGISTERED[alg] == s:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False,
                                               edgecolor=INK, linewidth=0.9))
        ax.axhline(len(PREDICTORS) - 0.5, color="white", linewidth=2.5)
        labels = [ALG_LABEL[a] for a in ALGORITHMS]
        if backbone == "resnet":
            labels[-1] = "DARE$^\\dagger$"
        ax.set_xticks(range(len(ALGORITHMS)), labels)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)), [SIG_LABEL[s] for s in rows] if k == 0 else [])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=7.5, pad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("Spearman $\\rho$", fontsize=6.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6, length=0)
    fig.savefig(OUT / "fig_signals.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "fig_signals.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dial(pd.read_csv(ROOT / "results/raw/merge_vit_full.csv"))
    fig_signals(pd.read_csv(ROOT / "results/analysis/signals.csv"))
    print(f"wrote {OUT}/fig_dial.pdf, {OUT}/fig_signals.pdf (previews in {PREVIEW})")


if __name__ == "__main__":
    main()
