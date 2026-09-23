"""Encoder-only fine-tuning with a frozen head.

Everything in this file is held identical across tasks, regimes and seeds --
optimizer, schedule, epochs, batch size, augmentation. The only thing that
varies between conditions is which task the encoder is shown, which is what
makes the comparison in the report a controlled one.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .data import TaskData, batches, to_model_input
from .evaluation import test_accuracy
from .models import BackboneSpec, TaskModel, freeze_batchnorm
from .utils import human_time


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    eval_batch_size: int = 256


@dataclass
class TrainResult:
    task: str
    probe_acc: float
    final_acc: float
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    """Linear warmup then cosine decay -- the schedule used in the course labs."""
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def finetune_task(
    model: TaskModel,
    task: TaskData,
    spec: BackboneSpec,
    device: torch.device,
    cfg: TrainConfig,
    probe_acc: float,
    seed: int = 0,
    verbose: bool = True,
) -> TrainResult:
    """Fine-tune ``model.encoder`` on ``task``; the head never moves."""
    model.to(device)
    params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    generator = torch.Generator().manual_seed(seed)
    steps_per_epoch = math.ceil(len(task.y_train) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    started = time.time()
    history: list[dict] = []
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        # Re-freeze after every train() call: the frozen head was fitted on
        # eval-mode features and is only valid while the encoder keeps
        # computing that same function. See models.freeze_batchnorm.
        freeze_batchnorm(model.encoder)
        running, seen = 0.0, 0
        for xb, yb in batches(
            task.x_train, task.y_train, cfg.batch_size, shuffle=True, generator=generator
        ):
            lr = _lr_at(step, total_steps, cfg.lr, cfg.warmup_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr

            xb = to_model_input(xb, device, spec.img_size, spec.mean, spec.std, train=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()

            running += loss.item() * yb.shape[0]
            seen += yb.shape[0]
            step += 1

        acc = test_accuracy(model, task, spec, device, cfg.eval_batch_size)
        history.append({"epoch": epoch, "train_loss": running / seen, "test_acc": acc})
        if verbose:
            print(
                f"    epoch {epoch + 1}/{cfg.epochs}  loss {running / seen:.4f}  "
                f"test acc {acc * 100:.2f}%",
                flush=True,
            )

    return TrainResult(
        task=task.name,
        probe_acc=probe_acc,
        final_acc=history[-1]["test_acc"],
        history=history,
        seconds=time.time() - started,
    )
