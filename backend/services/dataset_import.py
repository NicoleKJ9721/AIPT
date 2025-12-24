from pathlib import Path

from backend.models import Dataset

_DATASETS: list[Dataset] = []


def list_datasets() -> list[Dataset]:
    return _DATASETS


def register_dataset(name: str, root: Path) -> Dataset:
    dataset = Dataset(id=name, name=name, root=root, items=[])
    _DATASETS.append(dataset)
    return dataset
