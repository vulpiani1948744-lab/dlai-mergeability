#!/usr/bin/env python
"""Measure what input resolution costs us in accuracy, before committing to one.

Training at 224 is 3.4x slower than at 128 on this machine. That trade is only
worth paying if the lower resolution actually hurts, so we measure it: fit the
frozen linear probe on the pretrained encoder at each resolution and compare.
Feature extraction only -- no fine-tuning -- so this runs in a couple of minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergeability.data import build_tasks
from mergeability.models import build_backbone
from mergeability.probe import build_frozen_head
from mergeability.utils import get_device, human_time, save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="vit_tiny_patch16_224")
    p.add_argument("--resolutions", nargs="*", type=int, default=[64, 128, 224])
    p.add_argument("--regime", default="semantic")
    p.add_argument("--tasks", type=int, default=2)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", type=Path, default=Path("results/raw/resolution_check.json"))
    args = p.parse_args()

    device = get_device()
    tasks = build_tasks(args.regime, root=args.data_root)[: args.tasks]
    print(f"device={device}  backbone={args.backbone}  tasks={[t.name for t in tasks]}\n")

    rows = []
    print(f"{'res':>5}{'task':>24}{'probe acc':>12}{'time':>10}")
    print("-" * 51)
    for res in args.resolutions:
        encoder, spec = build_backbone(args.backbone, res, pretrained=True)
        encoder.to(device)
        for task in tasks:
            t0 = time.time()
            _, acc = build_frozen_head(encoder, task, spec, device, seed=0)
            elapsed = time.time() - t0
            rows.append({"resolution": res, "task": task.name, "probe_acc": acc})
            print(f"{res:>5}{task.name:>24}{acc * 100:>11.2f}%{human_time(elapsed):>10}")
        del encoder

    save_json({"backbone": args.backbone, "rows": rows}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
