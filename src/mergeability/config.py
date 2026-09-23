"""YAML experiment configs -> typed objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .finetune import TrainConfig


@dataclass
class BackboneConfig:
    name: str = "vit_tiny_patch16_224"
    img_size: int | None = None
    pretrained: bool = True


@dataclass
class DataConfig:
    root: str = "data"
    regimes: list[str] = field(default_factory=lambda: ["semantic", "random"])
    num_tasks: int = 5
    split_seed: int = 0


@dataclass
class OutputConfig:
    checkpoints: str = "checkpoints"
    results: str = "results/raw"


@dataclass
class ExperimentConfig:
    name: str = "benchmark_a"
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            name=raw.get("name", "benchmark_a"),
            seeds=raw.get("seeds", [0, 1]),
            backbone=BackboneConfig(**raw.get("backbone", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
