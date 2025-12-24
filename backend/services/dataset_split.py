import math
import random

from backend.models import Dataset, SplitConfig


def split_dataset(dataset: Dataset, config: SplitConfig) -> Dataset:
    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    items = list(dataset.items)
    random.Random(config.seed).shuffle(items)
    total = len(items)
    train_end = int(total * config.train_ratio)
    val_end = train_end + int(total * config.val_ratio)

    for index, item in enumerate(items):
        if index < train_end:
            item.split = "train"
        elif index < val_end:
            item.split = "val"
        else:
            item.split = "test"

    dataset.items = items
    return dataset
