import os
import datasets
import numpy as np
import tifffile

all_band_names = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12')
rgb_bands = ('B4', 'B3', 'B2')

S2_MEAN = [1354.40546513, 1118.24399958, 1042.92983953, 947.62620298, 1199.47283961, 1999.79090914, 2369.22292565, 2296.82608323, 732.08340178, 12.11327804, 1819.01027855, 1118.92391149, 2594.14080798]
S2_STD = [245.71762908, 333.00778264, 395.09249139, 593.75055589, 566.4170017, 861.18399006, 1086.63139075, 1117.98170791, 404.91978886, 4.77584468, 1002.58768311, 761.30323499, 1231.58581042]

class EuroSATConfig(datasets.BuilderConfig):
    BAND_SETS = {
        'all': all_band_names,
        'rgb': rgb_bands,
    }

    def __init__(self, bands=BAND_SETS['all'], **kwargs):
        super(EuroSATConfig, self).__init__(**kwargs)
        self.bands = bands
        self.band_indices = [int(all_band_names.index(b)) for b in bands if b in all_band_names]
    
    def get_config(self):
        return {'bands': self.bands}
    
    def __str__(self):
        return f"EuroSATConfig: bands={self.bands}, band_indices={self.band_indices}"

class EuroSATDataset(datasets.GeneratorBasedBuilder):
    """
    Config Args:
        - bands
        - band_indices: auto updated with bands, no need to manually entered to config
    """
    spatial_resolution = 10 
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
        EuroSATConfig(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    HEIGHT = WIDTH = 64

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        if config:
            if isinstance(config, dict):
                bands = config.get('bands')
            elif isinstance(config, EuroSATConfig):
                bands = config.get_config().get('bands')
            
            if bands:
                kwargs['bands'] = bands
                kwargs['band_indices'] = [all_band_names.index(b) for b in bands if b in all_band_names]

            
        self.height, self.width = self.HEIGHT, self.WIDTH
        self.labels = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake",
        ]

        super().__init__(*args, **kwargs)
        self.num_channels = len(self.config.band_indices)
        
        self.metadata["s2c"] = {
            key: [elem for i, elem in enumerate(value) 
                 if i in self.config.band_indices]
            for key, value in self.metadata["s2c"].items()
        }   

        print(self.config)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "radar": datasets.Array3D(shape=(2, self.height, self.width), dtype="float32"),  # Assuming radar has 2 channels (VV, VH)
                "optical": datasets.Array3D(shape=(len(self.config.band_indices), self.height, self.width), dtype="float32"),
                "label": datasets.ClassLabel(names=self.labels),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
                "radar_channel_wv": datasets.Sequence(datasets.Value("float32")),
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
                file_name = line.strip().replace(".jpg", ".tif")
                label = file_name.split('_')[0]
                data_path = os.path.join(data_dir, label, file_name)
                
                # read the image
                img = tifffile.imread(data_path)

                # permute img from HxWxC to CxHxW
                img = np.transpose(img, (2, 0, 1))
                # drop any channels if applicable
                img = np.take(img, indices=self.config.band_indices, axis=0)

                optical_channel_wv = np.array(self.metadata["s2c"]["channel_wv"])

                sample = {
                    "radar": None,
                    "optical": img.astype(np.float32),
                    "label": self.info.features['label'].str2int(label),
                    "optical_channel_wv": np.array(optical_channel_wv),
                    "radar_channel_wv": None,
                    "spatial_resolution": self.spatial_resolution,
                }

                yield f"{label}_{file_name}", sample
