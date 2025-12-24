import pytest

pytest.importorskip("pydantic")

from backend.models import Dataset, DatasetItem, SplitConfig
from backend.services.dataset_split import split_dataset


def test_split_dataset_assigns_splits() -> None:
    dataset = Dataset(
        id="d1",
        name="d1",
        root="/tmp",
        items=[
            DatasetItem(id="1", image_path="/tmp/1.png"),
            DatasetItem(id="2", image_path="/tmp/2.png"),
            DatasetItem(id="3", image_path="/tmp/3.png"),
            DatasetItem(id="4", image_path="/tmp/4.png"),
        ],
    )
    config = SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=1)
    result = split_dataset(dataset, config)
    splits = {item.split for item in result.items}
    assert splits == {"train", "val", "test"}


def test_split_dataset_ratio_validation() -> None:
    dataset = Dataset(id="d1", name="d1", root="/tmp", items=[])
    config = SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    with pytest.raises(ValueError):
        split_dataset(dataset, config)


def test_split_dataset_negative_ratio() -> None:
    dataset = Dataset(id="d1", name="d1", root="/tmp", items=[])
    config = SplitConfig(train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)
    with pytest.raises(ValueError):
        split_dataset(dataset, config)
