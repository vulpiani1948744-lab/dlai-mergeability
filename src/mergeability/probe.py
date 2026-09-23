"""Linear-probe-then-freeze heads.

Why this exists. If every task trained its own head jointly with its own
encoder, the head would be matched to *that* encoder. After merging, the
measured accuracy drop would then mix two different things: damage done by the
merge, and a head that no longer fits the encoder underneath it. The
task-arithmetic literature avoids this by using CLIP's frozen zero-shot heads.

We get the same property without paying for CLIP: fit each head on features from
the *pretrained* encoder, freeze it, and only then fine-tune the encoder. Every
model in the task-vector family, merged or not, is read out by the same head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .data import TaskData, batches, to_model_input
from .models import BackboneSpec


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    x_uint8: torch.Tensor,
    spec: BackboneSpec,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run the encoder over a uint8 image tensor and return CPU features."""
    encoder.eval()
    feats = []
    dummy = torch.zeros(x_uint8.shape[0], dtype=torch.long)
    for xb, _ in batches(x_uint8, dummy, batch_size, shuffle=False):
        xb = to_model_input(xb, device, spec.img_size, spec.mean, spec.std, train=False)
        feats.append(encoder(xb).float().cpu())
    return torch.cat(feats)


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
) -> nn.Linear:
    """Multinomial logistic regression on cached features, returned as a Linear.

    Fitting in closed form on cached features takes seconds, which is the whole
    point: the head costs nothing, so freezing it costs nothing either.
    """
    x = features.numpy().astype(np.float64)

    # Standardise before fitting: on raw ViT features lbfgs does not converge in
    # a reasonable number of iterations. The scaler is folded back into the
    # layer weights below, so the returned head still consumes raw features and
    # nothing downstream needs to know this happened.
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-6] = 1.0

    # scikit-learn >= 1.7 dropped the multi_class argument; lbfgs is multinomial
    # by default for multiclass targets, which is what we want here.
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, n_jobs=-1)
    clf.fit((x - mu) / sigma, labels.numpy())

    # logit = W((x - mu)/sigma) + b  =>  W' = W/sigma,  b' = b - (W mu)/sigma
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)
    if coef.shape[0] == 1 and num_classes == 2:  # sklearn's binary special case
        coef = np.vstack([-coef, coef]) / 2.0
        intercept = np.concatenate([-intercept, intercept]) / 2.0
    weight = coef / sigma
    bias = intercept - (coef * (mu / sigma)).sum(axis=1)

    head = nn.Linear(features.shape[1], num_classes)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
        head.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    head.requires_grad_(False)
    return head


def build_frozen_head(
    encoder: nn.Module,
    task: TaskData,
    spec: BackboneSpec,
    device: torch.device,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[nn.Linear, float]:
    """Fit and freeze a head for ``task`` on the pretrained encoder's features.

    Returns the head and its linear-probe test accuracy, which is the natural
    "before fine-tuning" reference point for that task.
    """
    train_feats = extract_features(encoder, task.x_train, spec, device, batch_size)
    head = fit_linear_probe(train_feats, task.y_train, task.num_classes, seed=seed)

    test_feats = extract_features(encoder, task.x_test, spec, device, batch_size)
    with torch.no_grad():
        preds = head(test_feats).argmax(dim=1)
    probe_acc = (preds == task.y_test).float().mean().item()
    return head, probe_acc
