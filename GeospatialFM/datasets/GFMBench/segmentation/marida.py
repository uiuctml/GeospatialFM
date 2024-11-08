import os
import datasets
import rasterio

import pandas as pd
import numpy as np

from PIL import Image
from itertools import product
from skimage import io

S2_MEAN = [0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443, 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]

S2_STD = [0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357, 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]

class MARIDAConfig(datasets.BuilderConfig):
    """BuilderConfig for MARIDA"""
    def __init__(self, data_dir=None, **kwargs):
        super(MARIDAConfig, self).__init__(**kwargs)
        self.data_dir = data_dir

    def get_config(self):
        config = {
            "data_dir": self.data_dir,
        }
        return config
    
    def __str__(self):
        return f"MARIDAConfig: data_dir={self.data_dir} \n"

class MARIDADataset(datasets.GeneratorBasedBuilder):
    spatial_resolution = 10 
    metadata = {
        "s2c": {
            "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4],
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
        MARIDAConfig(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    HEIGHT = WIDTH = 96

    patch_size = 96

    overlap = 16

    NUM_CLASSES = 11

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        self.num_channels = 11
        mean = np.array(S2_MEAN).astype(np.float32)
        self.impute_nan = np.tile(mean, (self.patch_size, self.patch_size, 1))

        assert config is not None, "config is required"
        data_dir = config.get_config().get('data_dir')
        kwargs['data_dir'] = data_dir

        super().__init__(*args, **kwargs)

        print(self.config)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "optical": datasets.Array3D(shape=(self.num_channels, self.HEIGHT, self.WIDTH), dtype="float32"),
                "label": datasets.Array2D(shape=(self.HEIGHT, self.WIDTH), dtype="int32"),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "spatial_resolution": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/MARIDA/resolve/main/MARIDA.zip")
        else:
            data_dir = os.path.join(self.config.data_dir, "MARIDA")

        split_file = os.path.join(data_dir, "metadata.csv")


        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "split_file": split_file,
                    "data_dir": os.path.join(data_dir, "patches"), 
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name="val",
                gen_kwargs={
                    "split_file": split_file,
                    "data_dir": os.path.join(data_dir, "patches"),
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "split_file": split_file,
                    "data_dir": os.path.join(data_dir, "patches"),
                    "split": "test",
                },
            )
        ]

    def _generate_examples(self, split_file, data_dir, split):
        optical_channel_wv = self.metadata["s2c"]["channel_wv"]

        metadata = pd.read_csv(split_file)
        metadata = metadata[metadata["split"] == split].reset_index(drop=True)

        for index, row in metadata.iterrows():
            roi_folder = '_'.join(['S2'] + row.filename.split('_')[:-1])
            roi_name = '_'.join(['S2'] + row.filename.split('_'))

            image_path = os.path.join(data_dir, roi_folder, roi_name + '.tif')
            target_path = os.path.join(data_dir, roi_folder, roi_name + '_cl.tif')

            img = rasterio.open(image_path)
            img_width, img_height = img.width, img.height       
            assert img_width == 256 and img_height == 256

            step_size = self.patch_size - self.overlap
            img_limits = product(
                range(0, img_height - self.patch_size + 1, step_size),
                range(0, img_width - self.patch_size + 1, step_size)
            )

            files = []
            for l in img_limits:
                files.append(
                    dict(limit=(l[0], l[1], l[0] + self.patch_size, l[1] + self.patch_size))
                )
            
            for i, file in enumerate(files):
                limit = file['limit']

                image, target = self.open_image(image_path, limit), self.open_image(target_path, limit).astype(np.int32)
                nan_mask = np.isnan(image)
                image[nan_mask] = self.impute_nan[nan_mask]
                target = target.astype(np.int32)
                target[target==15] = 7
                target[target==14] = 7
                target[target==13] = 7
                target[target==12] = 7
                target -= 1
                target[target==-1] = 255

                image = np.transpose(image, (2, 0, 1))

                sample = {
                    "optical": image,
                    "label": target,
                    "optical_channel_wv": optical_channel_wv,
                    "spatial_resolution": self.spatial_resolution
                }
                yield f"{index}_{i}", sample

    def open_image(self, img_path, limit=None):
        img = io.imread(img_path)
        img_cropped = img[limit[0]:limit[2], limit[1]:limit[3]]
        assert img_cropped.shape[0] == 96 and img_cropped.shape[1] == 96, f"dim0 is now having shape {img_cropped.shape[0]}, and dim1 is now having shape {img_cropped.shape[1]} instead of 96"
        return img_cropped.astype(np.float32)

# class MARIDADataset(Dataset):
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