import os
import glob
import json
import datasets
import tifffile
import pathlib
import rasterio

import numpy as np

from typing import Union
from rasterio.enums import Resampling
# from datasets import load_dataset
# from torch.utils.data import Dataset

Path = Union[str, pathlib.Path]

class_sets = {
        19: [
            'Urban fabric',
            'Industrial or commercial units',
            'Arable land',
            'Permanent crops',
            'Pastures',
            'Complex cultivation patterns',
            'Land principally occupied by agriculture, with significant areas of'
            ' natural vegetation',
            'Agro-forestry areas',
            'Broad-leaved forest',
            'Coniferous forest',
            'Mixed forest',
            'Natural grassland and sparsely vegetated areas',
            'Moors, heathland and sclerophyllous vegetation',
            'Transitional woodland, shrub',
            'Beaches, dunes, sands',
            'Inland wetlands',
            'Coastal wetlands',
            'Inland waters',
            'Marine waters',
        ],
        43: [
            'Continuous urban fabric',
            'Discontinuous urban fabric',
            'Industrial or commercial units',
            'Road and rail networks and associated land',
            'Port areas',
            'Airports',
            'Mineral extraction sites',
            'Dump sites',
            'Construction sites',
            'Green urban areas',
            'Sport and leisure facilities',
            'Non-irrigated arable land',
            'Permanently irrigated land',
            'Rice fields',
            'Vineyards',
            'Fruit trees and berry plantations',
            'Olive groves',
            'Pastures',
            'Annual crops associated with permanent crops',
            'Complex cultivation patterns',
            'Land principally occupied by agriculture, with significant areas of'
            ' natural vegetation',
            'Agro-forestry areas',
            'Broad-leaved forest',
            'Coniferous forest',
            'Mixed forest',
            'Natural grassland',
            'Moors and heathland',
            'Sclerophyllous vegetation',
            'Transitional woodland/shrub',
            'Beaches, dunes, sands',
            'Bare rock',
            'Sparsely vegetated areas',
            'Burnt areas',
            'Inland marshes',
            'Peatbogs',
            'Salt marshes',
            'Salines',
            'Intertidal flats',
            'Water courses',
            'Water bodies',
            'Coastal lagoons',
            'Estuaries',
            'Sea and ocean',
        ],
    }

label_converter = {
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

S2_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
S2_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

def sort_sentinel2_bands(x: Path) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split('_')[-1]
    x = os.path.splitext(x)[0]
    if x == 'B8A':
        x = 'B08A'
    return x

class BigEarthNetConfig(datasets.BuilderConfig):
    """BuilderConfig for BigEarthNet"""
    def __init__(self, data_dir=None, pad_s2=False, num_classes=19, **kwargs):
        assert num_classes in [43, 19], f"num_classes should be chosen from 43, 19, but got {num_classes}"
        super(BigEarthNetConfig, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.pad_s2 = pad_s2
        self.bands = "all"
        self.num_classes = num_classes

    def get_config(self):
        dict = {
            "data_dir": self.data_dir,
            "pad_s2": self.pad_s2,
            "num_classes": self.num_classes,
            "bands": self.bands,
        }
        return dict
    
    def __str__(self):
        return f"BigEarthNetConfig: data_dir={self.data_dir}, pad_s2={self.pad_s2}, bands={self.bands}, num_classes={self.num_classes} \n"

class BigEarthNetDataset(datasets.GeneratorBasedBuilder):
    spatial_resolution = 10 # TODO: not sure, make sure this is correct.
    metadata = {
        "s2c": {
            "bands":["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1613.7, 2202.4],
            "mean": S2_MEAN,
            "std": S2_STD
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
        BigEarthNetConfig(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    HEIGHT = WIDTH = 120

    NUM_CLASSES = 19

    def __init__(self, *args, **kwargs):
        self.height = self.HEIGHT
        self.width = self.WIDTH
        self.image_size = (self.height, self.width)
        self.optical_channels = 12
        self.radar_channels = 2
        self.class2idx = {c: i for i, c in enumerate(class_sets[43])}

        config = kwargs.pop('config', None)

        assert config is not None, "config is required"
        data_dir = config.get_config().get('data_dir')
        pad_s2 = config.get_config().get('pad_s2')
        num_classes = config.get_config().get('num_classes')
        kwargs['data_dir'] = data_dir
        kwargs['pad_s2'] = pad_s2
        kwargs['num_classes'] = num_classes
        self.NUM_CLASSES = num_classes

        if pad_s2:
            self.optical_channels += 1 # will pad B10 channel
            assert self.optical_channels == 13, "check code above again"

        super().__init__(*args, **kwargs)
        if self.config.pad_s2:
            self.metadata["s2c"]["bands"].insert(10, "B10")
            self.metadata["s2c"]["channel_wv"].insert(10, 1373.5)
            self.metadata["s2c"]["mean"].insert(10, 0.0)
            self.metadata["s2c"]["std"].insert(10, 0.0)
        print(self.config)

    def _info(self):
        features = {
            "optical": datasets.Array3D(shape=(self.optical_channels, self.height, self.width), dtype="float32"),
            "radar": datasets.Array3D(shape=(self.radar_channels, self.height, self.width), dtype="float32"),
            "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
            "radar_channel_wv": datasets.Sequence(datasets.Value("float32")),
            "label": datasets.Sequence(datasets.Value("float32"), length=self.config.num_classes),
            "spatial_resolution": datasets.Value("int32"),
        }

        return datasets.DatasetInfo(features=datasets.Features(features))

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/BigEarthNet/resolve/main/BigEarthNet.zip") # TODO: check out the correct address for downloading and extracting
        else:
            data_dir = os.path.join(self.config.data_dir, "BigEarthNet")
        radar_dir = os.path.join(data_dir, "BigEarthNet-S1-v1.0")
        image_dir = os.path.join(data_dir, "BigEarthNet-v1.0")
        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "bigearthnet-train.csv"),
                    "radar_dir": radar_dir,
                    "image_dir": image_dir, 
                },
            ),
            datasets.SplitGenerator(
                name="val",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "bigearthnet-val.csv"),
                    "radar_dir": radar_dir,
                    "image_dir": image_dir, 
                },
            ),
            datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "bigearthnet-test.csv"),
                    "radar_dir": radar_dir,
                    "image_dir": image_dir,  
                },
            ),
        ]

    def _generate_examples(self, split_file, radar_dir, image_dir):
        optical_channel_wv = self.metadata["s2c"]["channel_wv"]
        radar_channel_wv = self.metadata["s1"]["channel_wv"]

        self.folders = self._load_folders(split_file, radar_dir, image_dir)
        for files_folder in self.folders:
            image = self._load_image(files_folder)
            label = self._load_target(files_folder)

            radar, optical = image[:2], image[2:]
            if self.config.pad_s2:
                assert optical.shape[0] == 12
                optical = np.insert(optical, 10, np.zeros((self.height, self.width)), axis=0)                

            sample = {
                "optical": optical,
                "radar": radar,
                "label": label,
                "optical_channel_wv": np.array(optical_channel_wv),
                "radar_channel_wv": np.array(radar_channel_wv),
                "spatial_resolution": self.spatial_resolution,
            }

            yield f"{files_folder['s1']}/{files_folder['s2']}", sample

        

    def _load_paths(self, files_folder):
        folder_s1 = files_folder['s1'] 
        folder_s2 = files_folder['s2']
        paths_s1 = glob.glob(os.path.join(folder_s1, '*.tif')) 
        paths_s2 = glob.glob(os.path.join(folder_s2, '*.tif'))
        paths_s1 = sorted(paths_s1)
        paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
        paths = paths_s1 + paths_s2 

        return paths
    
    def _load_image(self, files_folder):
        paths = self._load_paths(files_folder)
        images = []
        if paths is not None:
            for path in paths:
                with rasterio.open(path) as dataset:
                    array = dataset.read(
                        indexes=1,
                        out_shape=self.image_size,
                        out_dtype='int32',
                        resampling=Resampling.bilinear,
                    )
                    images.append(array)
            arrays: np.typing.NDArray[np.int_] = np.stack(images, axis=0)

        return arrays
    
    def _load_target(self, files_folder):
        folder = files_folder['s2']
        path = glob.glob(os.path.join(folder, '*.json'))[0]
        with open(path) as f:
            labels = json.load(f)['labels'] # eg. "labels": ["Complex cultivation patterns", "Land principally occupied by agriculture, with significant areas of natural vegetation", "Broad-leaved forest"]
        indices = [self.class2idx[label] for label in labels]
        if self.config.num_classes == 19:
            indices_optional = [label_converter.get(idx) for idx in indices]
            indices = [idx for idx in indices_optional if idx is not None]
        target = np.zeros(self.config.num_classes, dtype=np.int64)
        target[indices] = 1
        return target
    
    def _load_folders(self, filename, dir_s1, dir_s2):
        with open(filename) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(',') for line in lines]

        folders = [
            {
                's1': os.path.join(dir_s1, pair[1]) if pair[1] != 'nan' else None,
                's2': os.path.join(dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders
    
# class BigEarthNetDataset(Dataset):
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