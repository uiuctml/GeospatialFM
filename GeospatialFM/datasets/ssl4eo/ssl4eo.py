import datasets
import numpy as np
import pandas as pd
import os
import rasterio
import random
import torch

_METADATA_URL = "metadata.csv"

# SSL4EO data statistics
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

class SSL4EODataset(datasets.GeneratorBasedBuilder):
    size = 264
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
            "channel_wv": [5500, 5700], # the wavelength is acutually 5.6 cm, which explodes the channel embedding, thus we use a dummy value
            "mean": S1_MEAN,
            "std": S1_STD
        }
    }

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "optical": datasets.Array3D(shape=(13, self.size, self.size), dtype="float32"),
                "radar": datasets.Array3D(shape=(2, self.size, self.size), dtype="float32"),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "radar_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "spatial_resolution": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={})
        ]

    def _generate_examples(self):
        root = self.config.data_dir
        metadata_path = os.path.join(root, _METADATA_URL)
        metadata = pd.read_csv(metadata_path)[:10]

        optical_bands = self.metadata["s2c"]["bands"]
        radar_bands = self.metadata["s1"]["bands"]
        optical_channel_wv = self.metadata["s2c"]["channel_wv"]
        radar_channel_wv = self.metadata["s1"]["channel_wv"]

        for key, row in enumerate(metadata.iterrows()):
            data = row[1]
            timestamp = random.randint(0, 3)
            
            optical_subdirs = data[f"s2c_t{timestamp}"]
            radar_subdirs = data[f"s1_t{timestamp}"]
            
            opticals = []
            optical_directory = os.path.join(root, optical_subdirs)
            for band in optical_bands:
                filename = os.path.join(optical_directory, f"{band}.tif")
                with rasterio.open(filename) as f:
                    image = f.read(out_shape=(1, self.size, self.size))
                    opticals.append(torch.from_numpy(image.astype(np.float32)))

            radars = []
            radar_directory = os.path.join(root, radar_subdirs)
            for band in radar_bands:
                filename = os.path.join(radar_directory, f"{band}.tif")
                with rasterio.open(filename) as f:
                    radar = f.read(out_shape=(1, self.size, self.size))
                    radars.append(torch.from_numpy(radar.astype(np.float32)))
            
            yield key, {
                "optical": np.concatenate(opticals),
                "radar": np.concatenate(radars),
                "optical_channel_wv": np.array(optical_channel_wv),
                "radar_channel_wv": np.array(radar_channel_wv),
                "spatial_resolution": self.spatial_resolution
            }