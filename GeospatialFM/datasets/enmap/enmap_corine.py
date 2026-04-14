import os
import random
from typing import Optional
from collections import defaultdict

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .enmap import S2C_MEAN, S2C_STD, S2C_WV


class EnMAPCorineDataset(Dataset):
    """PyTorch dataset for EnMAP-CORINE samples."""
    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    # Mapping from Corine codes to label indices
    corine_to_label_dict = {
        111: 0,
        112: 1,
        121: 2,
        122: 3,
        123: 4,
        124: 5,
        131: 6,
        132: 7,
        133: 8,
        141: 9,
        142: 10,
        211: 11,
        212: 12,
        213: 13,
        221: 14,
        222: 15,
        223: 16,
        231: 17,
        241: 18,
        242: 19,
        243: 20,
        244: 21,
        311: 22,
        312: 23,
        313: 24,
        321: 25,
        322: 26,
        323: 27,
        324: 28,
        331: 29,
        332: 30,
        333: 31,
        334: 32,
        411: 33,
        412: 34,
        421: 35,
        422: 36,
        423: 37,
        511: 38,
        512: 39,
        521: 40,
        522: 41,
        523: 42,
    }

    # Mapping from 43-class labels to 19-class labels
    label_converter_dict = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }
    corine_to_label = defaultdict(lambda: 43, corine_to_label_dict)
    label_converter = defaultdict(lambda: 43, label_converter_dict)

    num_classes = 19 


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
        "num_classes": 19,
        "ignore_index": 255,
    }

    image_root = "enmap"
    mask_root = "corine"

    def __init__(self, root: str, split: str, transform) -> None:
        """
        Args:
            root: Root directory containing the dataset.
            split: Optional split subdirectory inside ``root``.
            transform: Optional transform to be applied on a sample.
        """
        self.root = os.path.join(root, "enmap_corine")
        self.split_file = os.path.join(root, "splits", "enmap_corine", f"{split}.txt")
        self.split = split
        self.transform = transform
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file not found: {self.split_file}")

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
            mask = torch.from_numpy(src.read()).long()  # shape: (H, W)
            mask = mask.apply_(lambda x: self.corine_to_label[x])
            mask = mask.apply_(lambda x: self.label_converter[x])
            indices = torch.unique(mask)
            if indices[-1] == 43:
                indices = indices[:-1]  # Remove the default class
            label = torch.zeros(self.num_classes, dtype=torch.int)
            label[indices] = 1


        if self.transform is not None:
            optical, _, spatial_resolution = self.transform(
                optical=optical,
                radar=None,
                spatial_resolution=self.spatial_resolution
            )


        return {
            "optical": optical,
            "radar": None,
            "optical_channel_wv": self.metadata["s2c"]["channel_wv"],
            "radar_channel_wv": None,
            "spatial_resolution": spatial_resolution,
            "label": label,
        }

    def __len__(self) -> int:
        return len(self.sample_collection)
