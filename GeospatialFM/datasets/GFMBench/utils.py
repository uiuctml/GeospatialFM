from .classification import EuroSATConfig, BigEarthNetConfig, So2SatConfig, FMoWConfig, EuroSATDataset, BigEarthNetDataset, So2SatDataset, FMoWDataset
from .segmentation import SegMunichConfig, DFC2020Config, MARIDAConfig, SegMunichDataset, DFC2020Dataset, MARIDADataset

import os

DATASET_PATH = {
    "EuroSAT": "eurosat",
    "BigEarthNet": "bigearthnet",
    "So2Sat": "so2sat",
    "FMoW": "fmow",
    "SegMunich": "segmunich",
    "DFC2020": "dfc2020",
    "MARIDA": "marida",
}

DATASET = {
    "EuroSAT": EuroSATDataset,
    "BigEarthNet": BigEarthNetDataset,
    "So2Sat": So2SatDataset,
    "FMoW": FMoWDataset,
    "SegMunich": SegMunichDataset,
    "DFC2020": DFC2020Dataset,
    "MARIDA": MARIDADataset,
}

CONFIG = {
    "EuroSAT": EuroSATConfig,
    "BigEarthNet": BigEarthNetConfig,
    "So2Sat": So2SatConfig,
    "FMoW": FMoWConfig,
    "SegMunich": SegMunichConfig,
    "DFC2020": DFC2020Config,
    "MARIDA": MARIDAConfig,
}

def get_metadata(dataset_name):
    return DATASET[dataset_name].metadata

def get_dataset(args, train_transform, eval_transform):
    dataset_path = DATASET_PATH[args.dataset_name]
    dataset_path = os.path.join(args.data_dir, dataset_path)
    dataset_class = DATASET[args.dataset_name]
    config = CONFIG[args.dataset_name]() # TODO: what to pass in?
    
    dataset_dict = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else eval_transform
        dataset_dict[split] = dataset_class(root=dataset_path, split=split, config=config).with_transform(transform)
    return dataset_dict
