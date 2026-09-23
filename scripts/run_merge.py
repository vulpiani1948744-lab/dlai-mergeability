#!/usr/bin/env python
"""Merge every subset of tasks with every algorithm, and measure the damage.

The main experiment. Writes one CSV row per (regime, seed, subset, algorithm,
hyperparameters), carrying both the outcome -- normalized accuracy -- and every
pre-merge signal computed from the task vectors of that subset.

Nothing in the signals can see the merged model, so downstream they are
predictions and not descriptions.

    uv run scripts/run_merge.py --config configs/benchmark_a.yaml
    uv run scripts/run_merge.py --config configs/benchmark_a.yaml --sizes 2 --tasks 4
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergeability.config import ExperimentConfig
from mergeability.data import build_tasks
from mergeability.experiment import load_bundle, run_group
from mergeability.models import build_backbone
from mergeability.similarity import build_probe_set
from mergeability.utils import get_device, human_time, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--regimes", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--sizes", nargs="*", type=int, default=[2, 3, 5],
                   help="subset sizes to merge (default: pairs, triples, all)")
    p.add_argument("--tasks", type=int, default=None, help="use only the first N tasks")
    p.add_argument("--probe-per-task", type=int, default=400,
                   help="unlabelled probe images taken from each task's test split")
    p.add_argument("--out", type=Path, default=Path("results/raw/merge_results.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.regimes:
        cfg.data.regimes = args.regimes
    if args.seeds:
        cfg.seeds = args.seeds

    device = get_device(args.device)
    set_seed(0)

    encoder, spec = build_backbone(cfg.backbone.name, cfg.backbone.img_size, pretrained=False)
    encoder.to(device).eval()
    print(f"backbone={spec.name}  img_size={spec.img_size}  device={device}")
    print(f"subset sizes={tuple(args.sizes)}\n")

    started = time.time()
    rows: list[dict] = []

    for regime in cfg.data.regimes:
        tasks = build_tasks(regime, root=cfg.data.root, split_seed=cfg.data.split_seed,
                            num_tasks=cfg.data.num_tasks, download=False)
        if args.tasks:
            tasks = tasks[: args.tasks]
        probe_images = build_probe_set(tasks, per_task=args.probe_per_task)

        for seed in cfg.seeds:
            print(f"  [{regime}/seed{seed}]  {len(tasks)} tasks, "
                  f"probe set {len(probe_images)} images", flush=True)
            bundle = load_bundle(Path(cfg.output.checkpoints), spec.name, regime, seed, spec, tasks)
            rows += run_group(bundle, encoder, probe_images, device,
                              sizes=tuple(args.sizes), batch_size=cfg.train.eval_batch_size)
            print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:                      # union of keys, first-seen order
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}  ({human_time(time.time() - started)})")


if __name__ == "__main__":
    main()
