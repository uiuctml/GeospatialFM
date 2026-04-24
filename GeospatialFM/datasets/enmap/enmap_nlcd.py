import os
import random
from typing import Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets.nlcd import NLCD

from .enmap import S2C_MEAN, S2C_STD, S2C_WV


class EnMAPNLCDataset(Dataset):
    """PyTorch dataset for EnMAP-NLCD samples."""

    classes = [0, 11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    ignore_index = len(classes) - 1
    num_classes = len(classes) - 1  # excluding ignore_index 

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
    mask_root = "nlcd"

    def __init__(self, root: str, split: str, transform) -> None:
        """
        Args:
            root: Root directory containing the dataset.
            split: Optional split subdirectory inside ``root``.
            transform: Optional transform to be applied on a sample.
        """
        self.root = os.path.join(root, "enmap_nlcd")
        self.split_file = os.path.join(root, "splits", "enmap_nlcd", f"{split}.txt")
        self.split = split
        self.transform = transform
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        
        self.ordinal_map = torch.zeros(max(NLCD.cmap.keys()) + 1, dtype=torch.long) + len(self.classes) - 1
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        self.classes.remove(0)  
        self.classes.append(0)
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(NLCD.cmap[k])

        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file not found: {self.split_file}")

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
            mask = self.ordinal_map[mask]  # remap to ordinal labels

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
