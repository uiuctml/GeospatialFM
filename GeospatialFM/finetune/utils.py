from GeospatialFM.data import apply_transforms, pretrain_transform, multimodal_collate_fn
from GeospatialFM.datasets.utils import get_ssl4eo_metadata
from torch.utils.data import DataLoader
from functools import partial
from GeospatialFM.datasets import SSL4EODataset

import torch.nn as nn

class SpatialSpectralLowRankViTWithTaskHead(nn.Module):
    def __init__(self, encoder, task_head):
        super().__init__()
        self.encoder = encoder
        self.task_head = task_head

    def forward(self, x):
        x = self.encoder(x)
        x = self.task_head(x)
        return x

def get_task_head(model_config):
    pass

def get_dataloader(args):
    metadata = get_ssl4eo_metadata()
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    radar_mean, radar_std = metadata["s1"]["mean"], metadata["s1"]["std"]
    
    dataset = dict(train=SSL4EODataset(root=args.data_dir))
    standard_transform = partial(apply_transforms, optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std, use_8bit=args.use_8bit)
    collate_fn = partial(multimodal_collate_fn, transform=pretrain_transform, normalization=standard_transform)
    
    train_dataloader = DataLoader(
            dataset['train'],
            batch_size = args.train_batch_size,
            collate_fn=collate_fn,
            num_workers = args.dataloader_num_workers,
            pin_memory = args.dataloader_pin_memory,
            shuffle=True
        )
    
    return dataset, collate_fn, train_dataloader