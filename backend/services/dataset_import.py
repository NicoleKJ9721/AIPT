from pathlib import Path

from backend.models import Dataset, DatasetItem

_DATASETS: list[Dataset] = [
    Dataset(
        id="demo-dataset",
        name="demo-dataset",
        root=Path("/data/demo"),
        items=[
            DatasetItem(id="img-1", image_path=Path("/data/demo/img-1.png")),
            DatasetItem(id="img-2", image_path=Path("/data/demo/img-2.png")),
        ],
    )
]


def list_datasets() -> list[Dataset]:
    return _DATASETS


def register_dataset(name: str, root: Path) -> Dataset:
    dataset = Dataset(id=name, name=name, root=root, items=[])
    _DATASETS.append(dataset)
    return dataset
