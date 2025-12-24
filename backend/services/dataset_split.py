from backend.models import Dataset, SplitConfig


def split_dataset(dataset: Dataset, config: SplitConfig) -> Dataset:
    if config.train_ratio + config.val_ratio + config.test_ratio != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
    return dataset
