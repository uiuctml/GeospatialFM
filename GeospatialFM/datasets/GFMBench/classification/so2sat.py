import os
import h5py
import datasets
import numpy as np

means_per_version = {
    '2': 
        [
            -0.00003591224260,
            -0.00000765856128,
            0.00005937385750,
            0.00002516623150,
            0.04420110660000,
            0.25761027100000,
            0.00075567433700,
            0.00135034668000,
            0.12375696117681,
            0.10927746363683,
            0.10108552032678,
            0.11423986161140,
            0.15926566920230,
            0.18147236008771,
            0.17457403122913,
            0.19501607349635,
            0.15428468872076,
            0.10905050699570,
        ]
    ,
    '3_random':
        [
            -0.00005541164581,
            -0.00001363245448,
            0.00004558943283,
            0.00002990907940,
            0.04451951629749,
            0.25862310103671,
            0.00032720731137,
            0.00123416595462,
            0.12428656593186,
            0.11001677362564,
            0.10230652367417,
            0.11532195526186,
            0.15989486018315,
            0.18204406482475,
            0.17513562590622,
            0.19565546643221,
            0.15648722649020,
            0.11122536338577,
        ]
    ,
    '3_block': 
        [
            -0.00004632368791,
            0.00001260869365,
            0.00005305557337,
            0.00003471369557,
            0.04449937686171,
            0.26046026815721,
            0.00087815394475,
            0.00086889627435,
            0.12381869777901,
            0.10944155483024,
            0.10176911573221,
            0.11465267892206,
            0.15870528223797,
            0.18053964470203,
            0.17366821871719,
            0.19390983961551,
            0.15536490486611,
            0.11057334452833,
        ]
    ,
}
means_per_version['3_culture_10'] = means_per_version['2']

stds_per_version = {
    '2': 
        [
            0.17555201,
            0.17556463,
            0.45998793,
            0.45598876,
            2.85599092,
            8.32480061,
            2.44987574,
            1.46473530,
            0.03958795,
            0.04777826,
            0.06636616,
            0.06358874,
            0.07744387,
            0.09101635,
            0.09218466,
            0.10164581,
            0.09991773,
            0.08780632,
        ]
    ,
    '3_random': 
        [
            0.1756914,
            0.1761190,
            0.4600589,
            0.4563601,
            2.2492179,
            7.9056503,
            2.1917633,
            1.3148480,
            0.0392269,
            0.0470917,
            0.0653264,
            0.0624057,
            0.0758367,
            0.0891717,
            0.0905092,
            0.0996856,
            0.0990188,
            0.0873386,
        ]
    ,
    '3_block': 
        [
            0.1751797,
            0.1754073,
            0.4610124,
            0.4572122,
            0.8294254,
            7.1771026,
            0.9642598,
            0.8770835,
            0.0388311,
            0.0464986,
            0.0643833,
            0.0616141,
            0.0753004,
            0.0886178,
            0.0899500,
            0.0991759,
            0.0983276,
            0.0865943,
        ]
    ,
}

stds_per_version['3_culture_10'] = stds_per_version['2']

versions = ['2', '3_random', '3_block', '3_culture_10']

all_s1_band_names = (
    'S1_B1',
    'S1_B2',
    'S1_B3',
    'S1_B4',
    'S1_B5',
    'S1_B6',
    'S1_B7',
    'S1_B8',
)
all_s2_band_names = (
    'S2_B02',
    'S2_B03',
    'S2_B04',
    'S2_B05',
    'S2_B06',
    'S2_B07',
    'S2_B08',
    'S2_B8A',
    'S2_B11',
    'S2_B12',
)
all_band_names = all_s1_band_names + all_s2_band_names

all_s2_band_wv = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4]
all_s1_band_wv = [5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900] # dummy values

rgb_bands = ('S2_B04', 'S2_B03', 'S2_B02')

BAND_SETS = {
    'all': all_band_names,
    's1': all_s1_band_names,
    's2': all_s2_band_names,
    'rgb': rgb_bands,
}

class So2SatConfig(datasets.BuilderConfig):
    def __init__(self, data_dir=None, pad_s2=False, so2sat_version="3_culture_10", bands=BAND_SETS['s2'], **kwargs):
        super(So2SatConfig, self).__init__(**kwargs)
        assert so2sat_version in versions, "incorrect version"

        self.data_dir = data_dir
        self.pad_s2 = pad_s2
        self.so2sat_version = so2sat_version
        self.bands = bands

        self.output_s1_any = any(elem in all_s1_band_names for elem in self.bands)
        self.output_s2_any = any(elem in all_s2_band_names for elem in self.bands)
        if self.pad_s2:
            assert self.output_s2_any, f"need to output s2 channels in order to pad s2"
        self.s1_band_indices = np.array(
            [
                all_s1_band_names.index(b)
                for b in self.bands
                if b in all_s1_band_names
            ]
        ).astype(int)
        self.s1_band_names = [all_s1_band_names[i] for i in self.s1_band_indices]

        self.s2_band_indices = np.array(
            [
                all_s2_band_names.index(b)
                for b in self.bands
                if b in all_s2_band_names
            ]
        ).astype(int)
        self.s2_band_names = [all_s2_band_names[i] for i in self.s2_band_indices]
    
    def get_config(self):
        dict = {
            "data_dir": self.data_dir,
            "pad_s2": self.pad_s2,
            "so2sat_version": self.so2sat_version,
            "bands": self.bands,
            # extra below, don't really need them
            "output_s1_any": self.output_s1_any,
            "output_s2_any": self.output_s2_any,
            "s1_band_indices": self.s1_band_indices,
            "s1_band_names": self.s1_band_names,
            "s2_band_indices": self.s2_band_indices,
            "s2_band_names": self.s2_band_names,
        }
        return dict
    
    def __str__(self):
        return f"So2SatConfig: data_dir={self.data_dir}, pad_s2={self.pad_s2}, so2sat_version={self.so2sat_version}, bands={self.bands}, output_s1_any: {self.output_s1_any}, output_s2_any: {self.output_s2_any} \n"

class So2Sat(datasets.GeneratorBasedBuilder):
    """
    Config Args:
        - bands
        - band_indices: auto updated with bands, no need to manually entered to config
    """
    spatial_resolution = 10 # TODO: not sure, make sure this is correct.

    filenames_by_version = {
        '2': {
            'train': 'training.h5',
            'validation': 'validation.h5',
            'test': 'testing.h5',
        },
        '3_random': {'train': 'random/training.h5', 'test': 'random/testing.h5'},
        '3_block': {'train': 'block/training.h5', 'test': 'block/testing.h5'},
        '3_culture_10': {
            'train': 'culture_10/training.h5',
            'test': 'culture_10/testing.h5',
        },
    }
    
    BUILDER_CONFIGS = [
        So2SatConfig(
            name="default",
            description="Default configuration"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    HEIGHT = WIDTH = 32

    def __init__(self, *args, **kwargs):
        config = kwargs.pop('config', None)
        config_keywords = ['data_dir', 'so2sat_version', 'bands', 'pad_s2']
        self.num_s1_channels = 0 # by default, bands = BAND_SETS['s2]
        self.num_s2_channels = 10 
    
        if config and isinstance(config, dict):
            print("hi")
            for key, value in config.items():
                print(key)
                if key in config_keywords:
                    kwargs[key] = value
                    if key == 'bands':
                        self.num_s1_channels = sum(1 for elem in value if elem in all_s1_band_names)
                        self.num_s2_channels = sum(1 for elem in value if elem in all_s2_band_names)
        elif config and isinstance(config, So2SatConfig):
            configure = config.get_config()
            for key, value in configure.items():
                if key in config_keywords:
                    kwargs[key] = value
                    if key == 'bands':
                        self.num_s1_channels = sum(1 for elem in value if elem in all_s1_band_names)
                        self.num_s2_channels = sum(1 for elem in value if elem in all_s2_band_names)

        self.height = self.HEIGHT
        self.width = self.WIDTH

        super().__init__(*args, **kwargs)

        self.metadata = {}
        self.metadata["s2c"] = {
            "bands": self.config.s2_band_names,
            "channel_wv": [all_s2_band_wv[i] for i in self.config.s2_band_indices],
            "mean": [means_per_version[self.config.so2sat_version][i+8] for i in self.config.s2_band_indices], # add 8 to offset s1 mean and std
            "std": [stds_per_version[self.config.so2sat_version][i+8] for i in self.config.s2_band_indices],
        }
        self.metadata["s1"] = {
            "bands": self.config.s1_band_names,
            "channel_wv": [all_s1_band_wv[i] for i in self.config.s1_band_indices],
            "mean": [means_per_version[self.config.so2sat_version][i] for i in self.config.s1_band_indices],
            "std": [stds_per_version[self.config.so2sat_version][i] for i in self.config.s1_band_indices],
        }

        if self.config.pad_s2:
            self.metadata["s2c"]["bands"].insert(0, "S2_B01")
            self.metadata["s2c"]["bands"].insert(9, "S2_B09")
            self.metadata["s2c"]["bands"].insert(10, "S2_B10")
            self.metadata["s2c"]["channel_wv"].insert(0, 442.7)
            self.metadata["s2c"]["channel_wv"].insert(9, 945.1)
            self.metadata["s2c"]["channel_wv"].insert(10, 1373.5)
            self.metadata["s2c"]["mean"].insert(0, 0.0)
            self.metadata["s2c"]["mean"].insert(9, 0.0)
            self.metadata["s2c"]["mean"].insert(10, 0.0)
            self.metadata["s2c"]["std"].insert(0, 0.0)
            self.metadata["s2c"]["std"].insert(9, 0.0)
            self.metadata["s2c"]["std"].insert(10, 0.0)

        print(self.config)

    def _info(self):
        features = {
            "label": datasets.Value("float32"),
            "spatial_resolution": datasets.Value("int32"),
        }
        if self.config.output_s1_any:
            features.update({
                "radar": datasets.Array3D(shape=(self.num_s1_channels, self.height, self.width), dtype="float32"),
                "radar_channel_wv": datasets.Sequence(datasets.Value("float32")),
            })
            
        if self.config.output_s2_any:
            features.update({
                "optical": datasets.Array3D(shape=(self.num_s2_channels, self.height, self.width), dtype="float32"),
                "optical_channel_wv": datasets.Sequence(datasets.Value("float32")),
            })

        return datasets.DatasetInfo(
            features=datasets.Features(features)
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            data_dir = dl_manager.download_and_extract("https://huggingface.co/datasets/yuxuanw8/EuroSAT/resolve/main/So2Sat.zip")
        else:
            data_dir = os.path.join(self.config.data_dir, "So2Sat")

        train_dir = os.path.join(data_dir, self.filenames_by_version[self.config.so2sat_version]['train'])
        val_dir = os.path.join(data_dir, self.filenames_by_version[self.config.so2sat_version]['train'])
        test_dir = os.path.join(data_dir, self.filenames_by_version[self.config.so2sat_version]['test'])
        
        # generate train-val split
        with h5py.File(train_dir, 'r') as f:
            size = f['label'].shape[0]
            np.random.seed(42)
            indices = np.arange(size)
            np.random.shuffle(indices) 
            self.train_size = int(0.9 * size)
            self.val_size= size - self.train_size

            self.train_indices = indices[:self.train_size]
            self.val_indices = indices[self.train_size:]

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "data_dir": train_dir,
                    "indices": self.train_indices,
                },
            ),
            datasets.SplitGenerator(
                name="val",
                gen_kwargs={
                    "data_dir": val_dir,
                    "indices": self.val_indices,
                },
            ),
            datasets.SplitGenerator(
                name="test",
                gen_kwargs={
                    "data_dir": test_dir,
                },
            )
        ]

    def _generate_examples(self, data_dir, indices=None):
        optical_channel_wv = self.metadata["s2c"]["channel_wv"]
        radar_channel_wv = self.metadata["s1"]["channel_wv"]

        if indices is None:
            with h5py.File(data_dir, 'r') as f:
                indices = np.arange(f['label'].shape[0])
        with h5py.File(data_dir, "r") as f:
            for index in indices:
                s1 = f['sen1'][index].astype(np.float64)  # convert from <f8 to float64
                s1 = np.take(s1, indices=self.config.s1_band_indices, axis=2)
                s2 = f['sen2'][index].astype(np.float64)  # convert from <f8 to float64
                s2 = np.take(s2, indices=self.config.s2_band_indices, axis=2)

                # convert one-hot encoding to int64 then torch int
                label = f['label'][index].argmax()

                s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
                s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format

                if self.config.pad_s2:
                    img_size = s2.shape[1:]
                    s2 = np.concatenate((
                        np.zeros((1, *img_size)),
                        s2[:8],
                        np.zeros((2, *img_size)),
                        s2[8:]
                    ), axis=0)

                sample = {
                    'label': label,
                    'spatial_resolution': self.spatial_resolution,
                }
                if s1.shape[0] > 0:
                    sample['radar'] = s1
                    sample['radar_channel_wv'] = radar_channel_wv
                if s2.shape[0] > 0:
                    sample['optical'] = s2
                    sample['optical_channel_wv'] = optical_channel_wv

                yield f"{label}_{index}", sample