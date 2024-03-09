from GeospatialFM.data import *
from GeospatialFM.models.utils import ViTMMModel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from multiprocessing import Value
from .precision import get_autocast
import pickle
import os
from tqdm import tqdm

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

# def get_sampler(cfg):
#     return None

def get_data(cfg, ddp=False):
    train_ds, val_ds, test_ds = get_datasets(cfg['DATASET'])
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_shuffle = ddp is False
    batch_size = cfg['TRAINER']['per_device_train_batch_size'] 
    eval_batch_size = cfg['TRAINER']['per_device_eval_batch_size']
    num_workers = cfg['TRAINER']['dataloader_num_workers']
    pin_memory = cfg['TRAINER']['dataloader_pin_memory']
    drop_last = cfg['TRAINER']['dataloader_drop_last']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, sampler=train_sampler)
    train_dl.num_samples = len(train_ds)
    train_dl.num_batches = len(train_dl)
    if val_ds is None and test_ds is None:
        data = dict(
        train=DataInfo(train_dl, train_sampler),
        )
    else: 
        val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        val_dl.num_samples = len(val_ds)
        val_dl.num_batches = len(val_dl)
        test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_dl.num_samples = len(test_ds)
        test_dl.num_batches = len(test_dl)
        data = dict(
            train=DataInfo(train_dl, train_sampler),
            val=DataInfo(val_dl),
            test=DataInfo(test_dl),
        )
    return data

class DictDataset(Dataset):
    def __init__(self, dict_data, feature_key='img_feature'):
        self.dataset = dict_data
        self.num_samples = len(self.dataset['label'])
        self.feature_key = feature_key

    def __getitem__(self, index):
        feature = self.dataset[self.feature_key][index]
        label = self.dataset['label'][index]
        return dict(feature=feature, label=label)

    def __len__(self):
        return len(self.dataset['label'])
    
def extract_features(model, data, args, split='train', cache_path=None, chunk_size=20000):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    model.eval()
    features = dict(label=[], img_feature=[])
    dataloader = data.dataloader
    chunk_cnt = 0
    chunk_idx = 0
    pbar = tqdm(enumerate(dataloader), total=dataloader.num_batches)
    n_chunks = dataloader.num_samples // chunk_size
    pbar.set_description_str(f'Chunk {chunk_idx+1}/{n_chunks}: ')
    for i, batch in pbar:
        if isinstance(model, ViTMMModel):
            optical, radar = batch['image'], batch['radar']
            if args.finetune_modal == 'optical':
                optical = optical.to(device=device, non_blocking=True)
                radar = None
            elif args.finetune_modal == 'radar':
                radar = radar.to(device=device, non_blocking=True)
                optical = None
            images = (optical, radar)
        else:
            images = batch['image'] if args.finetune_modal == 'optical' else batch['radar']
            images = images.to(device=device, non_blocking=True)
        label = batch['label']
        with autocast() and torch.no_grad():
            model_out = model(images, return_dict=True)['cls_token'].detach().cpu()
        features['label'].append(label.detach().cpu())
        features['img_feature'].append(model_out)
        chunk_cnt += len(label)
        if cache_path is not None and chunk_cnt > chunk_size:
            features['label'] = torch.cat(features['label'], dim=0)
            features['img_feature'] = torch.cat(features['img_feature'], dim=0)
            cache_file = os.path.join(cache_path, f"{split}_{chunk_idx}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            features = dict(label=[], img_feature=[])
            chunk_cnt = 0
            chunk_idx += 1
            pbar.set_description_str(f'Chunk {chunk_idx+1}/{n_chunks}: ')
    features['label'] = torch.cat(features['label'], dim=0)
    features['img_feature'] = torch.cat(features['img_feature'], dim=0)
    if cache_path is not None:
        if chunk_idx > 0:
            cache_file = os.path.join(cache_path, f"{split}_{chunk_idx}.pkl")
        else:
            cache_file = os.path.join(cache_path, f"{split}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    return load_features(cache_path, split)

def load_features(cache_path, split):
    # list all files in cache_path containing split
    cache_files = [f for f in os.listdir(cache_path) if split in f and f.endswith('.pkl')]
    cache_files.sort()
    features = dict(label=[], img_feature=[])
    for f_name in cache_files:
        with open(os.path.join(cache_path, f_name), 'rb') as f:
            chunk_features = pickle.load(f)
            features['label'].append(chunk_features['label'])
            features['img_feature'].append(chunk_features['img_feature'])
    features['label'] = torch.cat(features['label'], dim=0)
    features['img_feature'] = torch.cat(features['img_feature'], dim=0)
    return features