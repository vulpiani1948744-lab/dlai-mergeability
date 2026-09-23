#!/usr/bin/env python
"""Integrity check over a directory of task vectors.

Run this before any analysis. Task vectors are produced across sessions and
sometimes across code versions, and a single file written by an older, buggy
version silently contaminates every downstream number while still loading
cleanly. This happened once already: before BatchNorm buffers were excluded,
ResNet-18 task vectors carried ||tau|| = 14414 against 2.7 for ViT-Tiny.

The check is deliberately blind to *which* files we suspect. It measures every
norm and flags whatever sits far from the median, so the criterion is the data
rather than our memory of which run produced what.

    uv run scripts/verify_task_vectors.py checkpoints/resnet18
    uv run scripts/verify_task_vectors.py checkpoints/resnet18 --delete
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


def tau_norm(path: Path) -> float:
    tau = torch.load(path, map_location="cpu")
    return torch.cat([v.float().reshape(-1) for v in tau.values()]).norm().item()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="directory to scan, recursively")
    p.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="flag a file whose norm is more than this many times the median, "
        "or less than the median divided by it (default: 10)",
    )
    p.add_argument("--delete", action="store_true", help="remove flagged files")
    args = p.parse_args()

    files = sorted(args.root.rglob("*_tau.pt"))
    if not files:
        print(f"no task vectors under {args.root}")
        return 1

    rows = [(tau_norm(f), f) for f in files]
    median = statistics.median(n for n, _ in rows)
    flagged = [f for n, f in rows if n > args.tolerance * median or n < median / args.tolerance]

    print(f"{len(rows)} task vectors under {args.root}")
    print(f"median ||tau|| = {median:.3f}   tolerance = x{args.tolerance:g}\n")
    for n, f in rows:
        mark = "   <-- OUTLIER" if f in flagged else ""
        print(f"{n:12.3f}  {f.relative_to(args.root)}{mark}")

    if not flagged:
        print(f"\nall {len(rows)} within tolerance")
        return 0

    print(f"\n{len(flagged)} outlier(s)")
    if args.delete:
        for f in flagged:
            f.unlink()
        print(f"deleted {len(flagged)}; re-run the fine-tuning to regenerate them")
        return 0

    print("re-run with --delete to remove them")
    return 1


if __name__ == "__main__":
    sys.exit(main())
