import os.path as osp
from torch.utils.data import DataLoader, Subset, ConcatDataset

import torchgeo.datasets as tgds
import torchgeo.datamodules as tgdm

from .tramsforms import *
from .datasets import *

MY_DATASETS = {
    "BigEarthNet": myBigEarthNet,
    "So2Sat": mySo2Sat,
    "OSCD": myOSCD,
    "SSL4EO": mySSL4EO
}

DATA_ROOT = './data'

# SSL4EO data statistics
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]

S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

def get_mean_std(data_cfg): # CHANGE
    if data_cfg['kwargs']['bands'] == 's1':
        return S1_MEAN, S1_STD
    elif data_cfg['name'] != 'OSCD' and data_cfg['kwargs']['bands'] == 'all':
        return S1_MEAN + S2A_MEAN, S1_STD + S2A_STD  # For now, only support BigEarthNet
    try:
        dm = getattr(tgdm, data_cfg['name']+'DataModule')
        if data_cfg['name'] == 'So2Sat':
            version = data_cfg['kwargs']['version']
            mean = dm.means_per_version[version]
            std = dm.stds_per_version[version]
            return mean, std
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

def get_datasets(data_cfg):
    print(f"Training Dataset: {data_cfg['name']}")
    data_cfg['kwargs']['root'] = osp.join(data_cfg['root'], data_cfg['name'])
    data_mean_std = get_mean_std(data_cfg)
    if data_cfg['task_type'] in ['classification', 'multilabel']:
        eval_transform = make_eval_transform(**data_cfg['eval_transforms'], mean_std=data_mean_std)
        train_transform = make_classification_train_transform(**data_cfg['train_transforms'], mean_std=data_mean_std) if data_cfg['use_train_transform'] else eval_transform
    if data_cfg['task_type'] in ['pretrain']:
        eval_transform = make_eval_transform(**data_cfg['eval_transforms'], mean_std=data_mean_std)
        train_transform = make_pretrain_transform(**data_cfg['train_transforms'], mean_std=data_mean_std) if data_cfg['use_train_transform'] else eval_transform
    elif data_cfg['task_type'] in ['change_detection']:
        eval_transform = make_segmentation_eval_transform(**data_cfg['eval_transforms'], mean_std=data_mean_std)
        train_transform = make_segmentation_train_transform(**data_cfg['train_transforms'], mean_std=data_mean_std) if data_cfg['use_train_transform'] else eval_transform
    train_dataset = get_dataset(data_cfg, split='train', transforms=train_transform)
    test_dataset = get_dataset(data_cfg, split='test', transforms=eval_transform)
    try:
        val_dataset = get_dataset(data_cfg, split='val', transforms=eval_transform)
    except:
        val_dataset = test_dataset
    if data_cfg['train_split'] == 'trainval' and val_dataset is not None:
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        val_dataset = test_dataset
    if data_cfg['train_frac'] < 1.0:
        train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:int(len(train_dataset)*data_cfg['train_frac'])])
    if data_cfg['val_frac'] < 1.0:
        val_dataset = Subset(val_dataset, torch.randperm(len(val_dataset))[:int(len(val_dataset)*data_cfg['val_frac'])])
    print(f"Train Set: {len(train_dataset)}\t Val Set: {len(val_dataset)}\t Test Set: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset