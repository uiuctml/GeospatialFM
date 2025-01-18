from datasets import load_dataset, get_dataset_infos
import json

import os

GFMBENCH_SCRIPTS_PATH = os.path.dirname(__file__)

DATASET_PATH = {
    "eurosat": "EuroSAT_hf",
    "bigearthnet": "BigEarthNet",
    "so2sat": "So2Sat",
    "dfc2020": "DFC2020",
    "segmunich": "SegMunich",
    "marida": "MARIDA",
    "landsat": "SSL4EOLBenchmark"
}

DATASET = {
    "eurosat": 'GFM-Bench/EuroSAT',
    "bigearthnet": "GFM-Bench/BigEarthNet",
    "so2sat": "GFM-Bench/So2Sat",
    "dfc2020": "GFM-Bench/DFC2020",
    "segmunich": "GFM-Bench/SegMunich",
    "marida": "GFM-Bench/MARIDA",
    "landsat": "GFM-Bench/SSL4EO-L-Benchmark"
}

def get_metadata(dataset_name):
    dataset = DATASET[dataset_name.lower()]
    infos = get_dataset_infos(dataset, trust_remote_code=True)
    return json.loads(infos['default'].description)

def get_dataset(args, train_transform, eval_transform):
    dataset_path = DATASET_PATH[args.dataset_name.lower()]
    dataset_path = os.path.join(args.data_dir, dataset_path)
    # data_class_path = DATASET_CLASS[args.dataset_name.lower()]
    # data_class_path = os.path.join(GFMBENCH_SCRIPTS_PATH, data_class_path)
    os.makedirs(dataset_path, exist_ok=True)
    # config = CONFIG[args.dataset_name.lower()](data_dir=args.data_dir) # TODO: what to pass in?
    dataset_name = DATASET[args.dataset_name.lower()]

    dataset_frac = {"train_frac": args.train_frac, "val_frac": args.val_frac, "test_frac": args.test_frac}

    dataset_dict = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else eval_transform
        dataset = load_dataset(dataset_name, split=split, cache_dir=dataset_path, trust_remote_code=True)
        split_frac = dataset_frac.get(f"{split}_frac")
        if split_frac != 1.0:
            dataset = dataset.train_test_split(train_size=split_frac, seed=42)['train'] 
        dataset_dict[split] = dataset.with_transform(transform)
        
    return dataset_dict

baseline_metadata = { # TODO: add your model's channels here
    "croma": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"], # oli_tirs_toa
    # "croma": ["B0", "B1", "B2", "B3", "B6L", "B6H", "B0", "B4", "B0", "B8", "B5", "B7"], # etm_toa
    "satmae": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    "spectralgpt": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"], 
    "scalemae": ["B4", "B3", "B2"],
    "satmae++": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
}

landsat_metadata = { # only work for oli_tirs_toa
    'croma': ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11", "B12", "B13"], # "B13" is just padding 
    'satmae': ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11", "B12"] # Remove B1 (coastal/Aerosol) and B9 (cirrus) from oli_tirs, add B12 as padding to make sure there are 10 channels
}

def get_baseline_metadata(args):
    baseline_name = args.model_name
    if args.dataset_name.lower().strip() == "landsat":
        return landsat_metadata[baseline_name]
    return baseline_metadata[baseline_name]