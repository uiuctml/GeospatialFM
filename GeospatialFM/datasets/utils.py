from .ssl4eo.ssl4eo_hf import SSL4EODataset
from .eurosat.eurosat import EuroSAT
from .bigearthnet.bigearthnet import BigEarthNet
from .so2sat.so2sat import So2Sat
from .fmow.fmow import FMoW

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

def get_fmow_metada(config=None):
    dataset = FMoW(config=config)
    return dataset.metadata