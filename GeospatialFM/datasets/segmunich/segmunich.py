import os
import datasets
import numpy as np
import tifffile

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

S2_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

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

class SegMunich(datasets.GeneratorBasedBuilder):
    spatial_resolution = 10 # TODO: not sure, make sure this is correct.
    metadata = {
        "s2c": {
            "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
            "channel_wv": [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1613.7, 2202.4],
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

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        config_keywords = ['data_dir', 'pad_s2']
        self.num_channels = 12
     
        if config:
            if isinstance(config, dict):
                for key, value in config.items():
                    if key in config_keywords:
                        kwargs[key] = value
                        if key == "pad_s2":
                            if value:
                                self.num_channels += 1
            elif isinstance(config, SegMunichConfig):
                configure = config.get_config()
                for key, value in configure.items():
                    if key in config_keywords:
                        kwargs[key] = value
                        if key == "pad_s2":
                            if value:
                                self.num_channels += 1

        super().__init__(*args, **kwargs)

        # modify metadata based on pad_s2
        if self.config.pad_s2:
            self.metadata["s2c"]["bands"].insert(10, "B10")
            self.metadata["s2c"]["channel_wv"].insert(10, 1373.5)
            self.metadata["s2c"]["mean"].insert(10, 0.0)
            self.metadata["s2c"]["std"].insert(10, 0.0)

        print(self.config)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "optical": datasets.Array3D(shape=(self.num_channels, self.HEIGHT, self.WIDTH), dtype="float32"),
                "label": datasets.ClassLabel(names=self.labels),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "spatial_resolution": datasets.Value("int32"),
            }),
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/EuroSAT/resolve/main/EuroSAT.zip") # TODO: check out the correct address for downloading and extracting

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "eurosat-train.txt"),
                    "data_dir": os.path.join(data_dir, "EuroSAT"), 
                },
            ),
            datasets.SplitGenerator(
                name="val",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "eurosat-val.txt"),
                    "data_dir": os.path.join(data_dir, "EuroSAT"),
                },
            ),
            datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "split_file": os.path.join(data_dir, "eurosat-test.txt"),
                    "data_dir": os.path.join(data_dir, "EuroSAT"),
                },
            )
        ]

    def _generate_examples(self, split_file, data_dir):

        """
        split_file: the filename that lists all data points
        data_dir: directory where the actual data is stored
        """

        """
        data_dir should be in following structure:
            - EuroSAT
                - AnnualCrop
                    - AnnualCrop_1.tif
                    - AnnualCrop_2.tif
                    - ...
                - Forest
                    - Forest_1.tif
                    - FOrest_2.tif
                    - ...
                - ...
        """

        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
        
                file_name = line.replace(".jpg", ".tif")
                label = file_name.split('_')[0]
                data_path = os.path.join(data_dir, label, file_name)
                
                img = tifffile.imread(data_path)

                # permute img from HxWxC to CxHxW
                img = np.transpose(img, (2, 0, 1))

                # drop any channels if applicable
                img = np.take(img, indices=self.config.band_indices, axis=0)

                optical_channel_wv = np.array(self.metadata["s2c"]["channel_wv"])
                optical_channel_wv = np.take(optical_channel_wv, indices=self.config.band_indices, axis=0)

                sample = {
                    "optical": img.astype(np.float32),
                    "label": self.info.features['label'].str2int(label),
                    "optical_channel_wv": np.array(optical_channel_wv),
                    "spatial_resolution": self.spatial_resolution,
                }

                yield f"{label}_{file_name}", sample