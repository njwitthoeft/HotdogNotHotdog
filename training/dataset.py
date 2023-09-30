"""Dataloader for very important proprietary hotdog dataset."""
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path

import random


class HotdogNotHotdogDataset(Dataset):
    """Dataset for proprietary awesome hotdog training data."""

    def __init__(self, hotdog_dir: Path, not_hotdog_dir: Path, seed=1) -> None:
        self.hotdog_dir = hotdog_dir
        self.not_hotdog_dir = not_hotdog_dir

        self.hotdog_paths = list(self.hotdog_dir.glob("*.jpg"))
        self.not_hotdog_paths = list(self.not_hotdog_dir.glob("*.jpg"))
        self.hotdog_len = len(self.hotdog_paths)
        self.not_hotdog_len = len(self.not_hotdog_paths)

        random.seed(seed)

        self.access = self.hotdog_paths + self.not_hotdog_paths
        self.access_ordering = list(range(self.hotdog_len + self.not_hotdog_len))
        random.shuffle(self.access_ordering)

    def __len__(self) -> int:
        return len(self.hotdog_paths) + len(self.not_hotdog_paths)

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        image_idx = self.access_ordering[index]
        label = int(image_idx < self.hotdog_len)  # "real" index is in the hotdog list
        image_path = self.access[image_idx]
        image = read_image(image_path)
        return image, label
