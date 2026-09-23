"""Accuracy of a (possibly merged) encoder read out by its frozen head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import TaskData, batches, to_model_input
from .models import BackboneSpec


@torch.no_grad()
def accuracy(
    model: nn.Module,
    x_uint8: torch.Tensor,
    y: torch.Tensor,
    spec: BackboneSpec,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    model.eval()
    correct = 0
    for xb, yb in batches(x_uint8, y, batch_size, shuffle=False):
        xb = to_model_input(xb, device, spec.img_size, spec.mean, spec.std, train=False)
        preds = model(xb).argmax(dim=1).cpu()
        correct += (preds == yb).sum().item()
    return correct / len(y)


def test_accuracy(
    model: nn.Module,
    task: TaskData,
    spec: BackboneSpec,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    return accuracy(model, task.x_test, task.y_test, spec, device, batch_size)
