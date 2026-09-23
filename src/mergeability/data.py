"""CIFAR-100 task families with a controlled inter-task similarity axis.

Benchmark A of the project. CIFAR-100 is partitioned into 5 disjoint 20-class
tasks under two regimes:

* ``semantic`` -- each task is the union of 4 related CIFAR-100 superclasses,
  so classes inside a task belong together and tasks are mutually distinct.
* ``random``   -- the 100 fine classes are shuffled and cut into 5 groups,
  so every task is an arbitrary mixture.

The two regimes are the single variable we change on purpose: they set the
inter-task similarity high or low while leaving everything else identical.

Each task has exactly 10,000 training and 2,000 test images (20 classes x
500/100 per class), so the training budget is uniform by construction and
dataset size is never a confound.

Images are kept in memory as uint8 tensors and resized/augmented on the
accelerator; this is several times faster than PIL-based transforms and is the
reason a full benchmark fits in an afternoon on a laptop GPU.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CIFAR100_URL_ROOT = "cifar-100-python"

# Five groups of four CIFAR-100 superclasses. Each superclass appears exactly
# once; the grouping is checked against the dataset metadata at load time.
SEMANTIC_GROUPS: dict[str, list[str]] = {
    "aquatic_life": [
        "aquatic_mammals",
        "fish",
        "reptiles",
        "non-insect_invertebrates",
    ],
    "flora_and_insects": [
        "flowers",
        "fruit_and_vegetables",
        "trees",
        "insects",
    ],
    "household_and_people": [
        "food_containers",
        "household_electrical_devices",
        "household_furniture",
        "people",
    ],
    "mammals": [
        "large_carnivores",
        "large_omnivores_and_herbivores",
        "medium_mammals",
        "small_mammals",
    ],
    "outdoor_and_vehicles": [
        "large_man-made_outdoor_things",
        "large_natural_outdoor_scenes",
        "vehicles_1",
        "vehicles_2",
    ],
}


@dataclass
class TaskData:
    """One task: a 20-class classification problem carved out of CIFAR-100."""

    name: str
    fine_classes: list[int]
    x_train: torch.Tensor  # uint8, (N, 3, 32, 32)
    y_train: torch.Tensor  # int64, remapped to 0..19
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def num_classes(self) -> int:
        return len(self.fine_classes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TaskData(name={self.name!r}, classes={self.num_classes}, "
            f"train={len(self.y_train)}, test={len(self.y_test)})"
        )


def _read_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    # keys come back as bytes on some versions; normalise to str
    return {(k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()}


def load_cifar100_raw(root: str | Path, download: bool = True) -> dict:
    """Load CIFAR-100 with both fine and coarse labels.

    torchvision's ``CIFAR100`` exposes only the fine labels, but the semantic
    regime needs the superclasses, so we let torchvision handle the download
    and then read the original pickles ourselves.
    """
    root = Path(root)
    base = root / CIFAR100_URL_ROOT
    if not base.exists():
        if not download:
            raise FileNotFoundError(f"CIFAR-100 not found at {base}")
        from torchvision.datasets import CIFAR100

        CIFAR100(root=str(root), train=True, download=True)

    train = _read_pickle(base / "train")
    test = _read_pickle(base / "test")
    meta = _read_pickle(base / "meta")

    def as_images(flat: np.ndarray) -> torch.Tensor:
        arr = np.asarray(flat, dtype=np.uint8).reshape(-1, 3, 32, 32)
        return torch.from_numpy(arr)

    return {
        "x_train": as_images(train["data"]),
        "fine_train": torch.tensor(train["fine_labels"], dtype=torch.long),
        "coarse_train": torch.tensor(train["coarse_labels"], dtype=torch.long),
        "x_test": as_images(test["data"]),
        "fine_test": torch.tensor(test["fine_labels"], dtype=torch.long),
        "coarse_test": torch.tensor(test["coarse_labels"], dtype=torch.long),
        "fine_names": list(meta["fine_label_names"]),
        "coarse_names": list(meta["coarse_label_names"]),
    }


def _semantic_task_classes(raw: dict) -> dict[str, list[int]]:
    """Map each semantic group to the 20 fine-class indices it contains."""
    coarse_names = raw["coarse_names"]
    grouped = {n for names in SEMANTIC_GROUPS.values() for n in names}
    missing = set(coarse_names) - grouped
    unknown = grouped - set(coarse_names)
    if missing or unknown:
        raise ValueError(
            f"SEMANTIC_GROUPS does not partition the CIFAR-100 superclasses "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )

    # fine class -> coarse class, read off the training labels
    fine_to_coarse: dict[int, int] = {}
    for fine, coarse in zip(raw["fine_train"].tolist(), raw["coarse_train"].tolist()):
        fine_to_coarse[fine] = coarse

    name_to_coarse = {n: i for i, n in enumerate(coarse_names)}
    tasks: dict[str, list[int]] = {}
    for task_name, superclasses in SEMANTIC_GROUPS.items():
        wanted = {name_to_coarse[s] for s in superclasses}
        tasks[task_name] = sorted(f for f, c in fine_to_coarse.items() if c in wanted)
    return tasks


def _random_task_classes(num_tasks: int, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(100)
    chunks = np.array_split(order, num_tasks)
    return {f"random_{i}": sorted(int(c) for c in chunk) for i, chunk in enumerate(chunks)}


def build_tasks(
    regime: str,
    root: str | Path = "data",
    split_seed: int = 0,
    num_tasks: int = 5,
    download: bool = True,
) -> list[TaskData]:
    """Build the 5 tasks of Benchmark A under the requested similarity regime.

    ``split_seed`` only controls the *random* regime; the semantic partition is
    fixed by the CIFAR-100 superclasses and does not depend on it.
    """
    raw = load_cifar100_raw(root, download=download)

    if regime == "semantic":
        task_classes = _semantic_task_classes(raw)
    elif regime == "random":
        task_classes = _random_task_classes(num_tasks, split_seed)
    else:
        raise ValueError(f"unknown regime {regime!r}, expected 'semantic' or 'random'")

    tasks: list[TaskData] = []
    for name, fine_classes in task_classes.items():
        remap = {c: i for i, c in enumerate(fine_classes)}
        out = {}
        for split in ("train", "test"):
            labels = raw[f"fine_{split}"]
            mask = torch.isin(labels, torch.tensor(fine_classes))
            idx = mask.nonzero(as_tuple=True)[0]
            out[f"x_{split}"] = raw[f"x_{split}"][idx].clone()
            out[f"y_{split}"] = torch.tensor(
                [remap[int(c)] for c in labels[idx].tolist()], dtype=torch.long
            )
        tasks.append(TaskData(name=name, fine_classes=fine_classes, **out))
    return tasks


# --------------------------------------------------------------------------
# batching and on-device preprocessing
# --------------------------------------------------------------------------

def batches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
):
    """Yield (x, y) minibatches from in-memory tensors."""
    n = x.shape[0]
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def _random_crop(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop with reflection padding, vectorised."""
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    left = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = top[:, None] + torch.arange(h, device=x.device)[None, :]
    cols = left[:, None] + torch.arange(w, device=x.device)[None, :]
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(c, device=x.device)[None, :, None, None]
    return x[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]


def to_model_input(
    x_uint8: torch.Tensor,
    device: torch.device,
    img_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    train: bool,
) -> torch.Tensor:
    """uint8 CIFAR batch -> normalised float batch at the backbone's resolution.

    Augmentation (random crop + horizontal flip) is applied at 32x32, before
    upsampling, so it costs almost nothing.
    """
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    if train:
        x = _random_crop(x, pad=4)
        flip = torch.rand(x.shape[0], device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if img_size != x.shape[-1]:
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s
