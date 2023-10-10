import os.path as osp
from torch.utils.data import DataLoader, Subset, ConcatDataset

import torchgeo.datasets as tgds
import torchgeo.datamodules as tgdm

from .tramsforms import make_classification_eval_transform, make_classification_train_transform, TransformSample
from .datasets import *

MY_DATASETS = {
    "BigEarthNet": myBigEarthNet,
    "So2Sat": mySo2Sat,
}

DATA_ROOT = './data'

def get_mean_std(data_cfg):
    try:
        dm = getattr(tgdm, data_cfg['name']+'DataModule')
        return dm.mean, dm.std
    except:
        return None, None

def get_dataset(data_cfg, split='train', transforms=None):
    if data_cfg['name'] in MY_DATASETS:
        return MY_DATASETS[data_cfg['name']](split=split, **data_cfg['kwargs'], transforms=transforms)
    try:
        return getattr(tgds, data_cfg['name'])(split=split, **data_cfg['kwargs'], transforms=transforms)
    except:
        raise NotImplementedError

def get_dataloaders(data_cfg):
    data_cfg['kwargs']['root'] = osp.join(data_cfg['root'], data_cfg['name'])
    train_dataset = get_dataset(data_cfg, split='train')
    val_dataset = get_dataset(data_cfg, split='val')
    test_dataset = get_dataset(data_cfg, split='test')
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, **data_cfg['dataloader'])
    val_dataloader = DataLoader(val_dataset, shuffle=False, **data_cfg['dataloader'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, **data_cfg['dataloader'])

    return train_dataloader, val_dataloader, test_dataloader

def get_datasets(data_cfg):
    data_cfg['kwargs']['root'] = osp.join(data_cfg['root'], data_cfg['name'])
    data_mean_std = get_mean_std(data_cfg)
    train_transform_func = make_classification_train_transform(**data_cfg['train_transforms'], mean_std=data_mean_std)
    eval_transform_func = make_classification_eval_transform(**data_cfg['eval_transforms'], mean_std=data_mean_std)
    eval_transform = TransformSample(eval_transform_func)
    train_transform = TransformSample(train_transform_func) if data_cfg['use_train_transform'] else eval_transform
    train_dataset = get_dataset(data_cfg, split='train', transforms=train_transform)
    val_dataset = get_dataset(data_cfg, split='val', transforms=eval_transform)
    test_dataset = get_dataset(data_cfg, split='test', transforms=eval_transform)
    if data_cfg['train_split'] == 'trainval':
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        val_dataset = test_dataset
    if data_cfg['train_frac'] < 1.0:
        train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:int(len(train_dataset)*data_cfg['train_frac'])])

    return train_dataset, val_dataset, test_dataset