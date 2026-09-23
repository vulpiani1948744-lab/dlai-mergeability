#!/usr/bin/env python
"""Fine-tune one encoder per task and store the resulting task vectors.

Week-1 entry point. For every (regime, seed, task) it

  1. fits a linear probe on the *pretrained* encoder's features and freezes it,
  2. fine-tunes the encoder only,
  3. writes tau = theta_ft - theta_0 to disk in fp16, plus the metrics.

Usage:
    uv run scripts/run_finetune.py --config configs/benchmark_a.yaml
    uv run scripts/run_finetune.py --config configs/smoke.yaml --limit-train 512
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergeability.config import ExperimentConfig
from mergeability.data import build_tasks
from mergeability.finetune import finetune_task
from mergeability.models import (
    TaskModel,
    build_backbone,
    encoder_param_dict,
    encoder_state_dict,
    freeze_batchnorm,
    task_vector,
)
from mergeability.probe import build_frozen_head
from mergeability.utils import get_device, human_time, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--regimes", nargs="*", default=None, help="override config regimes")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tasks", type=int, default=None, help="use only the first N tasks")
    p.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="subsample the training set (smoke tests only)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to idle between tasks; on a fanless machine sustained "
        "GPU load heat-soaks the chassis and throughput collapses",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.regimes:
        cfg.data.regimes = args.regimes
    if args.seeds:
        cfg.seeds = args.seeds
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    device = get_device(args.device)
    print(f"config={args.config}  device={device}")

    encoder0, spec = build_backbone(
        cfg.backbone.name, cfg.backbone.img_size, cfg.backbone.pretrained
    )
    print(
        f"backbone={spec.name}  img_size={spec.img_size}  "
        f"feature_dim={spec.feature_dim}  "
        f"params={sum(p.numel() for p in encoder0.parameters()) / 1e6:.2f}M"
    )
    encoder0.to(device)
    theta0 = encoder_state_dict(encoder0)      # full state, buffers included
    theta0_params = encoder_param_dict(encoder0)  # learnable parameters only
    n_bn = freeze_batchnorm(encoder0)
    if n_bn:
        print(f"froze {n_bn} BatchNorm layers (running statistics held fixed)")

    ckpt_root = Path(cfg.output.checkpoints) / spec.name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    torch.save(theta0, ckpt_root / "pretrained.pt")

    started = time.time()
    for regime in cfg.data.regimes:
        tasks = build_tasks(
            regime,
            root=cfg.data.root,
            split_seed=cfg.data.split_seed,
            num_tasks=cfg.data.num_tasks,
        )
        if args.tasks:
            tasks = tasks[: args.tasks]
        if args.limit_train:
            for t in tasks:
                t.x_train = t.x_train[: args.limit_train]
                t.y_train = t.y_train[: args.limit_train]

        for seed in cfg.seeds:
            out_dir = ckpt_root / regime / f"seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = {"regime": regime, "seed": seed, "backbone": spec.name, "tasks": {}}

            for task in tasks:
                tau_path = out_dir / f"{task.name}_tau.pt"
                if tau_path.exists() and not args.overwrite:
                    print(f"  [{regime}/seed{seed}] {task.name}: cached, skipping")
                    continue

                print(f"\n  [{regime}/seed{seed}] {task.name}  {task}")
                set_seed(seed)

                # Head is fitted on the PRETRAINED encoder, then frozen. See probe.py.
                encoder0.load_state_dict(theta0)
                encoder0.to(device)
                head, probe_acc = build_frozen_head(
                    encoder0, task, spec, device, cfg.train.eval_batch_size, seed=seed
                )
                print(f"    linear probe (theta_0) test acc {probe_acc * 100:.2f}%")

                encoder = copy.deepcopy(encoder0)
                model = TaskModel(encoder, head)
                result = finetune_task(
                    model, task, spec, device, cfg.train, probe_acc, seed=seed
                )

                theta_ft = encoder_param_dict(model)
                tau = task_vector(theta_ft, theta0_params)
                torch.save({k: v.half() for k, v in tau.items()}, tau_path)
                torch.save(
                    {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    out_dir / f"{task.name}_head.pt",
                )

                norm = torch.cat([v.reshape(-1) for v in tau.values()]).norm().item()
                summary["tasks"][task.name] = {
                    **result.to_dict(),
                    "tau_norm": norm,
                    "fine_classes": task.fine_classes,
                }
                print(
                    f"    done in {human_time(result.seconds)}  "
                    f"probe {probe_acc * 100:.2f}% -> finetuned {result.final_acc * 100:.2f}%  "
                    f"||tau||={norm:.2f}"
                )

                # Release the accelerator's cached blocks between tasks. Each
                # task builds a fresh encoder, so nothing here is reused.
                del model, encoder, theta_ft, tau
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

                if args.cooldown:
                    print(f"    cooling down for {args.cooldown:.0f}s")
                    time.sleep(args.cooldown)

            if summary["tasks"]:
                res_path = Path(cfg.output.results) / f"finetune_{spec.name}_{regime}_seed{seed}.json"
                save_json(summary, res_path)
                print(f"\n  wrote {res_path}")

    print(f"\ntotal {human_time(time.time() - started)}")


if __name__ == "__main__":
    main()
