import os
import torch
import datasets

import pandas as pd
import numpy as np
import skimage as io

from typing import ClassVar
from typing import Optional
from torchvision import transforms

all_band_names = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12')

rgb_bands = ('B4', 'B3', 'B2')

S2_MEAN = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433, 14.77112979, 1732.16362238, 1247.91870117]

S2_STD = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

class FMoWConfig(datasets.BuilderConfig):
    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': all_band_names,
        'rgb': rgb_bands,
    }
    """BuilderConfig for EuroSAT"""
    def __init__(self, data_dir=None, img_size=384, dropped_bands: Optional[list[str]] = None, dropped_bands_indices: Optional[list[int]] = None, **kwargs):
        super(FMoWConfig, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.img_size = img_size
        self.dropped_bands = dropped_bands
        self.dropped_bands_indices = dropped_bands_indices

    def get_config(self):
        config = {
            "data_dir": self.data_dir,
            "img_size": self.img_size,
            "dropped_bands": self.dropped_bands,
            "dropped_bands_indices": self.dropped_bands_indices,
        }
        return config

    def __str__(self):
        return f"FMoWConfig: data_dir={self.data_dir}, img_size={self.img_size}, dropped_bands={self.dropped_bands}, dropped_bands_indices={self.dropped_bands_indices} \n"

class FMoWDataset(datasets.GeneratorBasedBuilder):
    CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]

    spatial_resolution = 10 # TODO: not sure, make sure this is correct.
    metadata = {
        "s2c": {
            "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4],
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
        FMoWConfig(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        config_keywords = ['img_size', 'dropped_bands']
    
        self.height = 384
        self.width = 384
        if config:
            if isinstance(config, dict):
                for key, value in config.items():
                    if key in config_keywords:
                        kwargs[key] = value
                        if key == 'img_size':
                            self.height = value
                            self.width = value
                        elif key == 'dropped_bands':
                            kwargs['dropped_bands_indices'] = [int(all_band_names.index(b)) for b in kwargs['dropped_bands'] if b in all_band_names]
            elif isinstance(config, FMoWConfig):
                configure = config.get_config()
                for key, value in configure.items():
                    if key in config_keywords:
                        kwargs[key] = value
                        if key == 'img_size':
                            self.height = value
                            self.width = value
                        elif key == 'dropped_bands':
                            kwargs['dropped_bands_indices'] = [int(all_band_names.index(b)) for b in kwargs['dropped_bands'] if b in all_band_names]

        super().__init__(*args, **kwargs)

        # process self.metadata according to self.config.dropped_bands
        if self.config.dropped_bands is not None:
            for key in self.metadata["s2c"].keys():
                self.metadata["s2c"][key] = [elem for i, elem in enumerate(self.metadata["s2c"][key]) if i not in self.config.dropped_bands_indices]

        print(self.config)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "optical": datasets.Array3D(shape=(len(all_band_names) - len(self.config.dropped_bands), self.height, self.width), dtype="float32"),
                "label": datasets.Value("float32"),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "spatial_resolution": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/FMoW/resolve/main/FMoW.zip") # TODO: check out the correct address for downloading and extracting
        else:
            data_dir = os.path.join(self.config.data_dir, "FMoW")

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "train.csv"),
                    "data_dir": data_dir, 
                },
            ),
            datasets.SplitGenerator(
                name="val",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "val.csv"),
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "test.csv"),
                    "data_dir": data_dir,
                },
            )
        ]

    def _generate_examples(self, split_file, data_dir):
        def open_image(img_path):
            img = io.imread(img_path)
            return img.astype(np.float32)
        
        data = pd.read_csv(split_file).sort_values(['category', 'location_id', 'timestamp'])
        for row in data.itertuples(index=True, name='Pandas'):
            img_path = row.image_path
            image = open_image(os.path.join(data_dir, img_path)).transpose(2, 0, 1) # c, h, w

            label = self.CATEGORIES.index(row.category).float()
            
            if self.config.dropped_bands:
                keep_idxs = [i for i in range(image.shape[0]) if all_band_names[i] not in self.config.dropped_bands]
                image = image[keep_idxs, :, :] 
            
            # Resize image 
            image_tensor = torch.from_numpy(image)

            h, w = image.shape[1:]
            if h > w:
                w = self.config.img_size
            else:
                h = self.config.img_size
            
            # Resize the shorter side of image to 384
            image_tensor = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC)

            # Random crop to 384 x 384
            image_tensor = transforms.RandomCrop((h, w))
            image = image_tensor.numpy()

            print(image.shape)

            sample = {
                "optical": image,
                "label": label,
                "optical_channel_wv": self.metadata["s2c"]["channel_wv"],
                "spatial_resolution": self.spatial_resolution,
            }

            yield f"{label}_{row.index}", sample

