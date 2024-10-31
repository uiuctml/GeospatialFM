import os
import random
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

_METADATA_URL = "metadata.csv"

# SSL4EO data statistics
S1_MEAN = [-12.59, -20.26]
S1_STD = [5.26, 5.91]

S2C_MEAN = [1612.9, 1397.6, 1322.3, 1373.1, 1561.0, 2108.4, 2390.7, 2318.7, 2581.0, 837.7, 22.0, 2195.2, 1537.4]
S2C_STD = [791.0, 854.3, 878.7, 1144.9, 1127.5, 1164.2, 1276.0, 1249.5, 1345.9, 577.5, 47.5, 1340.0, 1142.9]

class SSL4EODataset(Dataset):
    spatial_resolution = 10
    metadata = {
        "s2c": {
            "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4],
            "mean": S2C_MEAN,
            "std": S2C_STD
        },
        "s1": {
            "bands": ["VV", "VH"],
            "channel_wv": [5500., 5700.],
            "mean": S1_MEAN,
            "std": S1_STD
        }
    }

    def __init__(self, root, metadata_url=_METADATA_URL):
        self.root = root
        self.size = 264
        # self.spatial_resolution = 10
        # self.metadata = {
        #     "s2c": {
        #         "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
        #         "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4],
        #         "mean": S2C_MEAN,
        #         "std": S2C_STD
        #     },
        #     "s1": {
        #         "bands": ["VV", "VH"],
        #         "channel_wv": [5500., 5700.],
        #         "mean": S1_MEAN,
        #         "std": S1_STD
        #     }
        # }
        
        metadata_path = os.path.join(root, metadata_url)
        self.metadata_df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        data = self.metadata_df.iloc[idx]
        timestamp = random.randint(0, 3)
        
        optical_subdirs = data[f"s2c_t{timestamp}"]
        radar_subdirs = data[f"s1_t{timestamp}"]
        
        opticals = []
        optical_directory = os.path.join(self.root, optical_subdirs)
        for band in self.metadata["s2c"]["bands"]:
            filename = os.path.join(optical_directory, f"{band}.tif")
            with rasterio.open(filename) as f:
                image = f.read(out_shape=(1, self.size, self.size))
                opticals.append(image.astype(np.float32))

        radars = []
        radar_directory = os.path.join(self.root, radar_subdirs)
        for band in self.metadata["s1"]["bands"]:
            filename = os.path.join(radar_directory, f"{band}.tif")
            with rasterio.open(filename) as f:
                radar = f.read(out_shape=(1, self.size, self.size))
                radars.append(radar.astype(np.float32))
        
        return {
            "optical": np.concatenate(opticals, axis=0),
            "radar": np.concatenate(radars, axis=0),
            "optical_channel_wv": self.metadata["s2c"]["channel_wv"],
            "radar_channel_wv": self.metadata["s1"]["channel_wv"],
            "spatial_resolution": self.spatial_resolution
        }