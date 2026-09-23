"""Backbones, frozen heads, and task-vector arithmetic.

The merging literature fine-tunes a shared pretrained encoder per task and
merges *encoders only*; the classification head is kept out of the merge. We
follow that convention, which is also what keeps the compared models
bit-identical in architecture -- there is no parameter-count asymmetry between
any two conditions in this study.
"""
from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

StateDict = dict[str, torch.Tensor]


@dataclass
class BackboneSpec:
    """Everything the rest of the code needs to know about a backbone."""

    name: str
    img_size: int
    feature_dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def build_backbone(
    name: str = "vit_tiny_patch16_224",
    img_size: int | None = None,
    pretrained: bool = True,
) -> tuple[nn.Module, BackboneSpec]:
    """Create a timm feature extractor (``num_classes=0``) and its data config.

    Normalisation statistics come from the backbone's own pretraining config,
    not from CIFAR: the encoder is pretrained, so its input distribution is the
    one it expects.
    """
    kwargs: dict = {"pretrained": pretrained, "num_classes": 0}
    if img_size is not None and "vit" in name:
        # ViTs need position embeddings resampled when the resolution changes;
        # timm does this for us when img_size is passed at creation time.
        kwargs["img_size"] = img_size

    model = timm.create_model(name, **kwargs)
    cfg = timm.data.resolve_model_data_config(model)
    resolved = img_size if img_size is not None else cfg["input_size"][-1]

    spec = BackboneSpec(
        name=name,
        img_size=int(resolved),
        feature_dim=int(model.num_features),
        mean=tuple(float(v) for v in cfg["mean"]),
        std=tuple(float(v) for v in cfg["std"]),
    )
    return model, spec


class TaskModel(nn.Module):
    """Encoder + a head that is frozen before fine-tuning ever starts.

    See ``probe.py`` for why the head is fitted on the *pretrained* features and
    then frozen: it makes the head shared across every encoder in the task-vector
    family, so a drop measured after merging is a property of the merge and not
    of a head that no longer matches its encoder.
    """

    def __init__(self, encoder: nn.Module, head: nn.Linear):
        super().__init__()
        self.encoder = encoder
        self.head = head
        for p in self.head.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def encoder_state_dict(model: nn.Module) -> StateDict:
    """Detached CPU copy of the full encoder state, buffers included."""
    enc = model.encoder if isinstance(model, TaskModel) else model
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def encoder_param_dict(model: nn.Module) -> StateDict:
    """Detached CPU copy of the encoder's *learnable parameters* only.

    Task vectors are built from this, never from the full state dict. A state
    dict also carries buffers -- for a ResNet, the BatchNorm ``running_mean``
    and ``running_var`` statistics -- which are accumulated averages of the
    data, not weights reached by gradient descent. Differencing them is
    meaningless, and because variances are unbounded it also swamps the norm:
    including them made ||tau|| read 14414 for ResNet-18 against 2.7 for
    ViT-Tiny, which has no buffers at all.
    """
    enc = model.encoder if isinstance(model, TaskModel) else model
    return {k: v.detach().cpu().clone() for k, v in enc.named_parameters()}


def freeze_batchnorm(module: nn.Module) -> int:
    """Hold every BatchNorm in eval mode; returns how many were frozen.

    BatchNorm normalises by batch statistics while training and by running
    statistics while evaluating, so the function the encoder computes changes
    the moment training starts. Our heads are linear probes fitted on the
    *pretrained encoder in eval mode* and then frozen, so that shift leaves the
    head reading a distribution it was never fitted on: ResNet-18 opened at a
    loss of 7.23 and ended fine-tuning 6 points *below* its own frozen probe.

    Freezing BatchNorm keeps the read-out function stable, which is what the
    linear-probe-then-freeze design requires. The affine weight and bias of each
    BatchNorm stay trainable and are part of the task vector; only the running
    statistics are held fixed.
    """
    frozen = 0
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            frozen += 1
    return frozen


# --------------------------------------------------------------------------
# task vectors
# --------------------------------------------------------------------------

def task_vector(finetuned: StateDict, pretrained: StateDict) -> StateDict:
    r"""$\tau = \theta_{\text{ft}} - \theta_0$, over float parameters only.

    Integer buffers (e.g. ``num_batches_tracked``) are skipped: they are counters,
    not parameters, and subtracting them is meaningless.
    """
    out: StateDict = {}
    for k, v in finetuned.items():
        if not torch.is_floating_point(v):
            continue
        out[k] = v.float() - pretrained[k].float()
    return out


def apply_task_vector(
    pretrained: StateDict, tau: StateDict, scaling: float = 1.0
) -> StateDict:
    r"""$\theta_0 + \lambda \tau$, leaving non-float buffers untouched."""
    out: StateDict = {k: v.clone() for k, v in pretrained.items()}
    for k, v in tau.items():
        out[k] = pretrained[k].float() + scaling * v
    return out


def flatten_state_dict(sd: StateDict, keys: list[str] | None = None) -> torch.Tensor:
    """Concatenate a state dict into one 1-D vector, in a fixed key order."""
    keys = sorted(sd) if keys is None else keys
    return torch.cat([sd[k].reshape(-1) for k in keys])


def mergeable_keys(tau: StateDict) -> list[str]:
    """Float keys shared by every task vector, in a deterministic order."""
    return sorted(tau)
