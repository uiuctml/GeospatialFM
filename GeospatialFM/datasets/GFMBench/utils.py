from .classification import EuroSAT, BigEarthNet, So2Sat, FMoW, EuroSATDataset, BigEarthNetDataset, So2SatDataset, FMoWDataset
from .segmentation import SegMunich, DFC2020, MARIDA, SegMunichDataset, DFC2020Dataset, MARIDADataset

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

METADATA = {
    "EuroSAT": EuroSAT.metadata,
    "BigEarthNet": BigEarthNet.metadata,
    "So2Sat": So2Sat.metadata,
    "FMoW": FMoW.metadata,
    "SegMunich": SegMunich.metadata,
    "DFC2020": DFC2020.metadata,
    "MARIDA": MARIDA.metadata,
}

def get_metadata(dataset_name):
    return METADATA[dataset_name]

def get_dataset(args, train_transform, eval_transform):
    dataset_path = DATASET_PATH[args.dataset_name]
    dataset_path = os.path.join(args.data_dir, dataset_path)
    dataset_class = DATASET[args.dataset_name]
    config = get_dataset_config(args)

    dataset_dict = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else eval_transform
        dataset_dict[split] = dataset_class(root=dataset_path, split=split, config=config).with_transform(transform)
    return dataset_dict

def get_dataset_config(args):
    pass