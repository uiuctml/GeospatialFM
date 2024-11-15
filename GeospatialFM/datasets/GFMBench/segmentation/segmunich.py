import os
import datasets
import tifffile

import pandas as pd
import numpy as np

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

all_band_names = [
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B09',
        'B11',
        'B12',
]

rgb_bands = ('B4', 'B3', 'B2')

S2_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2581.64687018, 2368.51236873, 1805.06846033]

S2_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1439.3086061, 1455.52084939, 1343.48379601]

class SegMunichConfig(datasets.BuilderConfig):
    """BuilderConfig for SegMunich"""
    def __init__(self, data_dir=None, pad_s2=False, **kwargs):
        super(SegMunichConfig, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.pad_s2 = pad_s2

    def get_config(self):
        config = {
            "data_dir": self.data_dir,
            "pad_s2": self.pad_s2,
        }

        return config
    
    def __str__(self):
        return f"SegMunichConfig: data_dir={self.data_dir}, pad_s2={self.pad_s2}\n"

class SegMunichDataset(datasets.GeneratorBasedBuilder):
    spatial_resolution = 10 # TODO: not sure, make sure this is correct.
    metadata = {
        "s2c": {
            "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 864.7, 1613.7, 2202.4],
            "mean": S2_MEAN,
            "std": S2_STD,
        },
        "s1": {
            "bands": None,
            "channel_wv": None,
            "mean": None,
            "std": None   
        }
    }

    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        SegMunichConfig(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    HEIGHT = WIDTH = 128

    NUM_CLASSES = 13

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        config_keywords = ['data_dir', 'pad_s2']
        self.num_channels = 10

        assert config is not None, "config is required"
        data_dir = config.get_config().get('data_dir')
        pad_s2 = config.get_config().get('pad_s2')
        kwargs['data_dir'] = data_dir
        kwargs['pad_s2'] = pad_s2

        if pad_s2:
            self.num_channels += 3

        super().__init__(*args, **kwargs)

        # modify metadata based on pad_s2
        if self.config.pad_s2:
            self.metadata["s2c"]["bands"].insert(7, "B8")
            self.metadata["s2c"]["channel_wv"].insert(7, 832.8)
            self.metadata["s2c"]["mean"].insert(7, 0.0)
            self.metadata["s2c"]["std"].insert(7, 0.0)

            self.metadata["s2c"]["bands"].insert(9, "B9")
            self.metadata["s2c"]["channel_wv"].insert(9, 945.1)
            self.metadata["s2c"]["mean"].insert(9, 0.0)
            self.metadata["s2c"]["std"].insert(9, 0.0)

            self.metadata["s2c"]["bands"].insert(10, "B10")
            self.metadata["s2c"]["channel_wv"].insert(10, 1373.5)
            self.metadata["s2c"]["mean"].insert(10, 0.0)
            self.metadata["s2c"]["std"].insert(10, 0.0)

        print(self.config)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "optical": datasets.Array3D(shape=(self.num_channels, self.HEIGHT, self.WIDTH), dtype="float32"),
                "label": datasets.Array2D(shape=(self.HEIGHT, self.WIDTH), dtype="float32"),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "spatial_resolution": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/SegMunich/resolve/main/SegMunich.zip")
        else:
            data_dir = os.path.join(self.config.data_dir, "SegMunich")

        split_file = os.path.join(data_dir, "dataset", "metadata.csv")
        metadata = pd.read_csv(split_file)

        counts = metadata['source'].value_counts()
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
        metadata = pd.read_csv(split_file)
        if indices is None:
            metadata = metadata[metadata['source'] == 'test'].reset_index(drop=True)
        else:
            metadata = metadata[metadata['source'] == 'train'].reset_index(drop=True)
            metadata = metadata.iloc[indices]

        for index, row in metadata.iterrows():
            file_name = row.id
            image_path = os.path.join(data_dir, "img", str(file_name) + ".tif")
            mask_path = os.path.join(data_dir, "label", str(file_name) + ".tif")

            img = tifffile.imread(image_path) # [128, 128, 10]
            img = np.transpose(img, (2, 0, 1)) #[10, 128, 128]
            target = np.array(Image.open(mask_path).convert("P")) # [128, 128]    
            target[target == 21] = 1 
            target[target == 22] = 2
            target[target == 23] = 3
            target[target == 31] = 4
            target[target == 32] = 6
            target[target == 33] = 7
            target[target == 41] = 8
            target[target == 13] = 9
            target[target == 14] = 10

            sample = {
                "optical": img,
                "label": target,
                "optical_channel_wv": self.metadata["s2c"]["channel_wv"],
                "spatial_resolution": self.spatial_resolution,
            }

            yield index, sample

# class SegMunichDataset(Dataset):
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