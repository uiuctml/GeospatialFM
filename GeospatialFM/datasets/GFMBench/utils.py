from .classification import EuroSATConfig, BigEarthNetConfig, So2SatConfig, FMoWConfig, EuroSATDataset, BigEarthNetDataset, So2SatDataset, FMoWDataset
from .segmentation import SegMunichConfig, DFC2020Config, MARIDAConfig, SegMunichDataset, DFC2020Dataset, MARIDADataset
from datasets import load_dataset

import os

GFMBENCH_SCRIPTS_PATH = os.path.dirname(__file__)

DATASET_PATH = {
    "eurosat": "EuroSAT_hf",
    "bigearthnet": "BigEarthNet_hf",
    "so2sat": "So2Sat_hf",
    "fmow": "FMoW_hf",
    "segmunich": "SegMunich_hf",
    "dfc2020": "DFC2020_hf",
    "marida": "MARIDA_hf",
}

DATASET = {
    "eurosat": EuroSATDataset,
    "bigearthnet": BigEarthNetDataset,
    "so2sat": So2SatDataset,
    "fmow": FMoWDataset,
    "segmunich": SegMunichDataset,
    "dfc2020": DFC2020Dataset,
    "marida": MARIDADataset,
}

CONFIG = {
    "eurosat": EuroSATConfig,
    "bigearthnet": BigEarthNetConfig,
    "so2sat": So2SatConfig,
    "fmow": FMoWConfig,
    "segmunich": SegMunichConfig,
    "dfc2020": DFC2020Config,
    "marida": MARIDAConfig,
}

DATASET_CLASS_PATH = {
    "eurosat": "classification/eurosat.py",
    "bigearthnet": "classification/bigearthnet.py",
    "so2sat": "classification/so2sat.py",
    "fmow": "classification/fmow.py",
    "segmunich": "segmentation/segmunich.py",
    "dfc2020": "segmentation/dfc2020.py",
    "marida": "segmentation/marida.py",
}

def get_metadata(dataset_name):
    metadata = DATASET[dataset_name.lower()].metadata
    metadata['size'] = min(DATASET[dataset_name.lower()].HEIGHT, DATASET[dataset_name.lower()].WIDTH)
    metadata['num_classes'] = DATASET[dataset_name.lower()].NUM_CLASSES
    return metadata

def get_dataset(args, train_transform, eval_transform):
    dataset_path = DATASET_PATH[args.dataset_name.lower()]
    dataset_path = os.path.join(args.data_dir, dataset_path)
    data_class_path = DATASET_CLASS_PATH[args.dataset_name.lower()]
    data_class_path = os.path.join(GFMBENCH_SCRIPTS_PATH, data_class_path)
    os.makedirs(dataset_path, exist_ok=True)
    config = CONFIG[args.dataset_name.lower()](data_dir=args.data_dir) # TODO: what to pass in?

    dataset_frac = {"train_frac": args.train_frac, "val_frac": args.val_frac, "test_frac": args.test_frac}

    dataset_dict = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else eval_transform
        dataset = load_dataset(path=data_class_path, split=split, config=config, cache_dir=dataset_path)
        split_frac = dataset_frac.get(f"{split}_frac")
        if split_frac != 1.0:
            dataset = dataset.train_test_split(train_size=split_frac, seed=args.seed)['train'] 
        dataset_dict[split] = dataset.with_transform(transform)
        
    return dataset_dict
