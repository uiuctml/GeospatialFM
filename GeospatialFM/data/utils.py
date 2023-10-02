import os.path as osp
from torch.utils.data import DataLoader

from torchgeo.datasets import BigEarthNet, SEN12MS, So2Sat, ETCI2021, EuroSAT

from .tramsforms import make_classification_eval_transform, make_classification_train_transform, TransformSample

DATA_ROOT = './data'

DATASET_DICT = {
    'BigEarthNet': BigEarthNet,
    'SEN12MS': SEN12MS,
    'So2Sat': So2Sat,
    'ETCI2021': ETCI2021,
    'EuroSAT': EuroSAT,
}

def get_dataset(data_cfg, split='train', transforms=None):
    return DATASET_DICT[data_cfg['name']](split=split, **data_cfg['kwargs'], transforms=transforms)

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
    train_transform_func = make_classification_train_transform(**data_cfg['train_transforms'])
    eval_transform_func = make_classification_eval_transform(**data_cfg['eval_transforms'])
    train_transform = TransformSample(train_transform_func)
    eval_transform = TransformSample(eval_transform_func)
    train_dataset = get_dataset(data_cfg, split='train', transforms=train_transform)
    val_dataset = get_dataset(data_cfg, split='val', transforms=eval_transform)
    test_dataset = get_dataset(data_cfg, split='test', transforms=eval_transform)

    return train_dataset, val_dataset, test_dataset