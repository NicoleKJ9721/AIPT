from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]


@dataclass
class DatasetItem:
    id: str
    image_path: Path
    split: SplitName | None = None
    label_path: Path | None = None


@dataclass
class Dataset:
    id: str
    name: str
    root: Path
    items: list[DatasetItem]


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int = 42
