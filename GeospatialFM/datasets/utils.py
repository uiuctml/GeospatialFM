from .ssl4eo.ssl4eo_hf import SSL4EODataset
from .eurosat.eurosat import EuroSAT, EuroSATDataset
from .bigearthnet.bigearthnet import BigEarthNet, BigEarthNetDataset
from .so2sat.so2sat import So2Sat, So2SatDataset
from .fmow.fmow import FMoW, FMoWDataset
from .segmunich.segmunich import SegMunich, SegMunichDataset
from .dfc2020.dfc2020 import DFC2020, DFC2020Dataset
from .marida.marida import MARIDA, MARIDADataset

def prepare_dataset_config(args): # TODO
    if args.dataset_name == "EuroSAT":
        # bands
        pass
    elif args.dataset_name == "BigEarthNet":
        pass
    elif args.dataset_name == "So2Sat":
        pass
    elif args.dataset_name == "FMoW":
        pass
    elif args.dataset_name == "SegMunich":
        pass
    elif args.dataset_name == "DFC2020":
        pass
    elif args.dataset_name == "MARIDA":
        pass
    else:
        raise NotImplementedError

def get_ssl4eo_metadata():
    dataset = SSL4EODataset()
    return dataset.metadata

def get_eurosat_metadata(config=None): 
    dataset = EuroSAT(config=config)
    return dataset.metadata

def get_ben_metadata(config=None):
    dataset = BigEarthNet(config=config)
    return dataset.metadata

def get_so2sat_metadata(config=None):
    dataset = So2Sat(config=config)
    return dataset.metadata

def get_fmow_metadata(config=None):
    dataset = FMoW(config=config)
    return dataset.metadata

def get_segmunich_metadata(config=None):
    dataset = SegMunich(config=config)
    return dataset.metadata

def get_dfc2020_metadata(config=None):
    dataset = DFC2020(config=config)
    return dataset.metadata

def get_marida_metadata(config=None):
    dataset = MARIDA(config=config)
    return dataset.metadata

def get_dataset(args, config):
    if args.dataset_name == "EuroSAT":
        train_dataset = EuroSATDataset(root="GeospatialFM/datasets/eurosat", split="train", config=config)
        eval_dataset = EuroSATDataset(root="GeospatialFM/datasets/eurosat", split="val", config=config)
        test_dataset = EuroSATDataset(root="GeospatialFM/datasets/eurosat", split="test", config=config)
        metadata = get_eurosat_metadata(config)
    elif args.dataset_name == "BigEarthNet":
        train_dataset = BigEarthNetDataset(root="GeospatialFM/datasets/bigearthnet", split="train", config=config)
        eval_dataset = BigEarthNetDataset(root="GeospatialFM/datasets/bigearthnet", split="val", config=config)
        test_dataset = BigEarthNetDataset(root="GeospatialFM/datasets/bigearthnet", split="test", config=config)
        metadata = get_ben_metadata(config)
    elif args.dataset_name == "So2Sat":
        train_dataset = So2SatDataset(root="GeospatialFM/datasets/so2sat", split="train", config=config)
        eval_dataset = So2SatDataset(root="GeospatialFM/datasets/so2sat", split="val", config=config)
        test_dataset = So2SatDataset(root="GeospatialFM/datasets/so2sat", split="test", config=config)
        metadata = get_so2sat_metadata(config)
    elif args.dataset_name == "FMoW":
        train_dataset = FMoWDataset(root="GeospatialFM/datasets/fmow", split="train", config=config)
        eval_dataset = FMoWDataset(root="GeospatialFM/datasets/fmow", split="val", config=config)
        test_dataset = FMoWDataset(root="GeospatialFM/datasets/fmow", split="test", config=config)
        metadata = get_fmow_metadata(config)
    elif args.dataset_name == "SegMunich":
        train_dataset = SegMunichDataset(root="GeospatialFM/datasets/segmunich", split="train", config=config)
        eval_dataset = SegMunichDataset(root="GeospatialFM/datasets/segmunich", split="val", config=config)
        test_dataset = SegMunichDataset(root="GeospatialFM/datasets/segmunich", split="test", config=config)
        metadata = get_segmunich_metadata(config)
    elif args.dataset_name == "DFC2020":
        train_dataset = DFC2020Dataset(root="GeospatialFM/datasets/dfc2020", split="train", config=config)
        eval_dataset = DFC2020Dataset(root="GeospatialFM/datasets/dfc2020", split="val", config=config)
        test_dataset = DFC2020Dataset(root="GeospatialFM/datasets/dfc2020", split="test", config=config)
        return get_dfc2020_metadata(config)
    elif args.dataset_name == "MARIDA":
        train_dataset = MARIDADataset(root="GeospatialFM/datasets/marida", split="train", config=config)
        eval_dataset = MARIDADataset(root="GeospatialFM/datasets/marida", split="val", config=config)
        test_dataset = MARIDADataset(root="GeospatialFM/datasets/marida", split="test", config=config)
        return get_marida_metadata(config)
    else:
        raise NotImplementedError
    
    dataset_dict = {
        "train": train_dataset,
        "val": eval_dataset,
        "test": test_dataset,
    }

    return dataset_dict, metadata