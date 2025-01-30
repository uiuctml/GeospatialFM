from datasets import load_dataset, get_dataset_infos
import json

import os

GFMBENCH_SCRIPTS_PATH = os.path.dirname(__file__)

DATASET_PATH = {
    "eurosat": "EuroSAT",
    "bigearthnet": "BigEarthNet",
    "dfc2020": "DFC2020",
    "segmunich": "SegMunich",
    "so2sat": "So2Sat",
    "marida": "MARIDA",
    "landsat": "SSL4EOLBenchmark",
}

DATASET = {
    "eurosat": 'GFM-Bench/EuroSAT',
    "bigearthnet": "GFM-Bench/BigEarthNet",
    "dfc2020": "GFM-Bench/DFC2020",
    "segmunich": "GFM-Bench/SegMunich",
    "so2sat": "GFM-Bench/So2Sat",
    "marida": "GFM-Bench/MARIDA",
    "landsat": "GFM-Bench/SSL4EO-L-Benchmark",
}

def get_metadata(dataset_name, dataset_version=None):
    dataset = DATASET[dataset_name.lower()]
    config_name = dataset_version if dataset_version else "default"
    infos = get_dataset_infos(dataset, trust_remote_code=True)
    return json.loads(infos[config_name].description)

def get_dataset(args, train_transform, eval_transform):
    dataset_path = DATASET_PATH[args.dataset_name.lower()]
    dataset_path = os.path.join(args.data_dir, dataset_path)
    # data_class_path = DATASET_CLASS[args.dataset_name.lower()]
    # data_class_path = os.path.join(GFMBENCH_SCRIPTS_PATH, data_class_path)
    os.makedirs(dataset_path, exist_ok=True)
    # config = CONFIG[args.dataset_name.lower()](data_dir=args.data_dir) # TODO: what to pass in?
    config_name = args.dataset_version
    dataset_name = DATASET[args.dataset_name.lower()]

    dataset_frac = {"train_frac": args.train_frac, "val_frac": args.val_frac, "test_frac": args.test_frac}

    dataset_dict = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else eval_transform
        dataset = load_dataset(dataset_name, split=split, cache_dir=dataset_path, trust_remote_code=True, name=config_name)
        split_frac = dataset_frac.get(f"{split}_frac")
        if split_frac != 1.0:
            dataset = dataset.train_test_split(train_size=split_frac, seed=42)['train'] 
        dataset_dict[split] = dataset.with_transform(transform)
        
    return dataset_dict
