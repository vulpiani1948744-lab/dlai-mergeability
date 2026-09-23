r"""Centered Kernel Alignment between the representations of two encoders.

Weight-space signals ask whether two models moved the same way. CKA asks
something different and, for our purposes, complementary: whether they ended up
*computing* the same thing. Two encoders can reach very different weights and
still produce representations that are rotations of one another -- weight-space
distance would call them far apart, CKA would call them identical.

Linear CKA (Kornblith et al., ICML 2019), for centred feature matrices
$X \in \mathbb{R}^{n \times d_1}$, $Y \in \mathbb{R}^{n \times d_2}$:

.. math::
    \mathrm{CKA}(X, Y) =
        \frac{\lVert Y^\top X \rVert_F^2}
             {\lVert X^\top X \rVert_F \, \lVert Y^\top Y \rVert_F}

It is invariant to orthogonal transformations and to isotropic scaling, which
is exactly what makes it usable across independently fine-tuned models, and it
does not require the two representations to have the same width.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import TaskData
from ..models import BackboneSpec
from ..probe import extract_features


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two (n_examples, n_features) matrices.

    The two may have different numbers of features but must describe the same
    examples, in the same order.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row count differs: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("need at least two examples")

    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)

    cross = (y.T @ x).norm() ** 2
    self_x = (x.T @ x).norm()
    self_y = (y.T @ y).norm()
    denom = self_x * self_y
    if denom == 0:
        return float("nan")
    return (cross / denom).item()


@torch.no_grad()
def encoder_cka(
    encoder_a: nn.Module,
    encoder_b: nn.Module,
    probe_images: torch.Tensor,
    spec: BackboneSpec,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """CKA between two encoders, measured on a shared probe set.

    The probe set is fixed and shared across every comparison in the study, so
    the numbers are on a common footing. It is unlabelled: CKA compares
    representations, not predictions.
    """
    fa = extract_features(encoder_a, probe_images, spec, device, batch_size)
    fb = extract_features(encoder_b, probe_images, spec, device, batch_size)
    return linear_cka(fa, fb)


def build_probe_set(tasks: list[TaskData], per_task: int = 400) -> torch.Tensor:
    """A fixed, balanced set of unlabelled images drawn from every task.

    Taken from the *test* splits so no probe image was trained on, and balanced
    across tasks so the measurement is not dominated by whichever task happens
    to be most familiar to one of the encoders.
    """
    chunks = [t.x_test[:per_task] for t in tasks]
    return torch.cat(chunks)
