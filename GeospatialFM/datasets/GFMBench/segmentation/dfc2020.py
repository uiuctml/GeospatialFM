import os
import datasets
import tifffile
import rasterio

import pandas as pd
import numpy as np
from skimage import io

from PIL import Image
from itertools import product
from datasets import load_dataset
from torch.utils.data import Dataset

S2_MEAN = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392, 1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783, 1972.62420416,  582.72633433, 14.77112979, 1732.16362238, 1247.91870117]

S2_STD = [633.15169573, 650.2842772, 712.12507725, 965.23119807, 948.9819932, 1108.06650639, 1258.36394548, 1233.1492281, 1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

S1_MEAN = [-12.54847273, -20.19237134]

S1_STD = [5.25697717, 5.91150917]

class DFC2020Config(datasets.BuilderConfig):
    """BuilderConfig for DFC2020"""
    def __init__(self, data_dir=None, **kwargs):
        super(DFC2020Config, self).__init__(**kwargs)
        self.data_dir = data_dir

    def get_config(self):
        config = {
            "data_dir": self.data_dir,
        }

        return config
    
    def __str__(self):
        return f"DFC2020Config: data_dir={self.data_dir}\n"

class DFC2020Dataset(datasets.GeneratorBasedBuilder):
    spatial_resolution = 10 
    metadata = {
        "s2c": {
            "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4],
            "mean": S2_MEAN,
            "std": S2_STD,
        },
        "s1": {
            "bands": ["VV", "VH"],
            "channel_wv": [5500, 5700],
            "mean": S1_MEAN,
            "std": S1_STD   
        }
    }

    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        DFC2020Config(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    HEIGHT = WIDTH = 96

    patch_size = 96

    overlap = 16

    DFC2020_CLASSES = [
        0,  # class 0 unused in both schemes
        1, 1, 1, 1, 1,
        2, 2,
        3,  # --> will be masked if no_savanna == True
        3,  # --> will be masked if no_savanna == True
        4,
        5,
        6,  # 12 --> 6
        7,  # 13 --> 7
        6,  # 14 --> 6
        8,
        9,
        10
    ]

    NUM_CLASSES = 8

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        self.optical_num_channels = 13
        self.radar_num_channels = 2
     
        assert config is not None, "config is required"
        data_dir = config.get_config().get('data_dir')
        kwargs['data_dir'] = data_dir

        super().__init__(*args, **kwargs)

        print(self.config)


    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "optical": datasets.Array3D(shape=(self.optical_num_channels, self.HEIGHT, self.WIDTH), dtype="float32"),
                "radar": datasets.Array3D(shape=(self.radar_num_channels, self.HEIGHT, self.WIDTH), dtype="float32"),
                "label": datasets.Array2D(shape=(self.HEIGHT, self.WIDTH), dtype="int32"),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "radar_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "spatial_resolution": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/DFC2020/resolve/main/DFC2020.zip")
        else:
            data_dir = os.path.join(self.config.data_dir, "DFC2020")

        split_file = os.path.join(data_dir, "metadata.csv")
        metadata = pd.read_csv(split_file)

        counts = metadata['split'].value_counts()
        size = counts.get("train")
        np.random.seed(42)
        indices = np.arange(size)
        np.random.shuffle(indices)
        self.train_size = int(0.9 * size)
        self.val_size = size - self.train_size

        self.train_indices = indices[:self.train_size]
        self.val_indices = indices[self.train_size:]

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "split_file": split_file,
                    "data_dir": os.path.join(data_dir, "train"), 
                    "indices": self.train_indices,
                },
            ),
            datasets.SplitGenerator(
                name="val",
                gen_kwargs={
                    "split_file": split_file,
                    "data_dir": os.path.join(data_dir, "train"),
                    "indices": self.val_indices,
                },
            ),
            datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "split_file": split_file,
                    "data_dir": os.path.join(data_dir, "test"),
                    "indices": None,
                },
            )
        ]

    def _generate_examples(self, split_file, data_dir, indices=None):
        optical_channel_wv = self.metadata["s2c"]["channel_wv"]
        radar_channel_wv = self.metadata["s1"]["channel_wv"]

        metadata = pd.read_csv(split_file)
        if indices is None:
            metadata = metadata[metadata['split'] == 'test'].reset_index(drop=True)
        else:
            metadata = metadata[metadata['split'] == 'train'].reset_index(drop=True)
            metadata = metadata.iloc[indices]

        for index, row in metadata.iterrows():
            # 'lc_path', 's1_path', 's2_path', 'split'
            # img = rasterio.open(row.s2_path)
            img = rasterio.open(os.path.join(data_dir, row.s2_path))
            img_width, img_height = img.width, img.height
            assert img_width == 256 and img_height == 256

            step_size = self.patch_size - self.overlap
            img_limits = product(
                range(0, img_height - self.patch_size + 1, step_size), 
                range(0, img_width - self.patch_size + 1, step_size)
            )

            files = []
            for l in img_limits:
                files.append(dict(limit=(l[0], l[1], l[0] + self.patch_size, l[1] + self.patch_size)))

            for i, file in enumerate(files):
                limit = file['limit']
                lc_path = os.path.join(data_dir, row.lc_path)
                s1_path = os.path.join(data_dir, row.s1_path)
                s2_path = os.path.join(data_dir, row.s2_path)
                optical, radar, target = self.open_image(img_path=s2_path, limit=limit), self.open_image(img_path=s1_path, limit=limit), self.open_image(img_path=lc_path, limit=limit).astype(np.int32)

                target = target[:, :, 0]
                target = np.take(self.DFC2020_CLASSES, target.astype(np.int64)) 
                target = target.astype(np.int32)
                target[target==8] = 0
                target[target > 8] -= 1
                target[target==3] = 0
                target[target > 3] -= 1
                target -= 1
                target[target == -1] = 255

                optical = np.transpose(optical, (2, 0, 1))
                radar = np.transpose(radar, (2, 0, 1))

                sample = {
                    "optical": optical,
                    "radar": radar,
                    "label": target,
                    "optical_channel_wv": optical_channel_wv,
                    "radar_channel_wv": radar_channel_wv,
                    "spatial_resolution": self.spatial_resolution,
                }

                yield f"{index}_{i}", sample

    def open_image(self, img_path, limit=None):
        img = io.imread(img_path)
        img_cropped = img[limit[0]:limit[2], limit[1]:limit[3]]
        assert img_cropped.shape[0] == 96 and img_cropped.shape[1] == 96, f"dim0 is now having shape {img_cropped.shape[0]}, and dim1 is now having shape {img_cropped.shape[1]} instead of 96"
        return img_cropped.astype(np.float32)

# class DFC2020Dataset(Dataset):
#     """
#     Wrapper class
#     """
#     def __init__(self, root, split="train", config=None):
#         super().__init__()
#         self.data = load_dataset(root, split=split, config=config, trust_remote_code=True)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]