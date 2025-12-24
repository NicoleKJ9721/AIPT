from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SplitName = Literal["train", "val", "test"]


class DatasetItem(BaseModel):
    id: str
    image_path: Path
    split: SplitName | None = None
    label_path: Path | None = None


class Dataset(BaseModel):
    id: str
    name: str
    root: Path
    items: list[DatasetItem] = Field(default_factory=list)


class SplitConfig(BaseModel):
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int = 42


class TrainingJob(BaseModel):
    id: str
    model: str
    status: str
