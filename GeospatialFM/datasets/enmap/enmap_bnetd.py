import os
import random
from typing import Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .enmap import S2C_MEAN, S2C_STD, S2C_WV


class EnMAPBNETDDataset(Dataset):
    """PyTorch dataset for EnMAP-BDForet samples."""

    foreground_classes = [0, 1, 2, 3, 6, 9, 10, 13, 15, 16, 21]
    ignore_index = len(foreground_classes)
    num_classes = len(foreground_classes)  # excluding ignore_index 

    spatial_resolution = 30
    metadata = {
        "s2c": {
            "bands": None,
            "channel_wv": S2C_WV,
            "mean": S2C_MEAN,
            "std": S2C_STD,
        },
        "s1": {
            "bands": None,
            "channel_wv": None,
            "mean": None,
            "std": None,
        },
        "num_classes": num_classes,
        "ignore_index": ignore_index,
    }

    image_root = "enmap"
    mask_root = "bnetd"

    def __init__(self, root: str, split: str, transform) -> None:
        """
        Args:
            root: Root directory containing the dataset.
            split: Optional split subdirectory inside ``root``.
            transform: Optional transform to be applied on a sample.
        """
        self.root = os.path.join(root, "enmap_bnetd")
        self.split_file = os.path.join(root, "splits", "enmap_bnetd", f"{split}.txt")
        self.split = split
        self.transform = transform
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file not found: {self.split_file}")

        self.mapping = {code: idx for idx, code in enumerate(self.foreground_classes)}
        # print(f"ignore_index: {self.ignore_index}")

    def read_split_file(self):
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines()]
        sample_collection = [
            (
                os.path.join(self.root, self.image_root, sample_id),
                os.path.join(self.root, self.mask_root, sample_id)
            )
            for sample_id in sample_ids
        ]
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, object]:
        img_path, mask_path = self.sample_collection[index]
        with rasterio.open(img_path) as src:
            optical = torch.from_numpy(src.read()).float()
        
        with rasterio.open(mask_path) as src:
            mask = torch.from_numpy(src.read()).long().squeeze(0)  # shape: (H, W)
            new_mask = torch.full_like(mask, fill_value=self.ignore_index)
            for code, ordinal in self.mapping.items():
                new_mask[mask == code] = ordinal
            mask = new_mask  # shape: (H, W)

        if self.transform is not None:
            optical, _, mask, spatial_resolution = self.transform(
                optical=optical,
                radar=None,
                label=mask,
                spatial_resolution=self.spatial_resolution
            )


        return {
            "optical": optical,
            "radar": None,
            "optical_channel_wv": self.metadata["s2c"]["channel_wv"],
            "radar_channel_wv": None,
            "spatial_resolution": spatial_resolution,
            "label": mask,
        }

    def __len__(self) -> int:
        return len(self.sample_collection)
