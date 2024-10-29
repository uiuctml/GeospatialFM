from GeospatialFM.data import apply_transforms, pretrain_transform, multimodal_collate_fn, DataCollator, train_classification_transform, eval_classification_transform, train_segmentation_transform, eval_segmentation_transform
from GeospatialFM.datasets.utils import get_ssl4eo_metadata, get_dataset, prepare_dataset_config
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

def get_task_head(model_config): # TODO
    pass

def get_dataloader(args):
    dataset_config = prepare_dataset_config(args)
    dataset, metadata = get_dataset(args=args, config=dataset_config)
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    radar_mean, radar_std = metadata["s1"]["mean"], metadata["s1"]["std"]

    standard_transform = partial(apply_transforms, optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std, use_8bit=args.use_8bit)
    
    assert args.task_type in ["classification", "multilabel", "segmentation"], f"invalid task type: {args.task_type}"
    train_transform = train_classification_transform if args.task_type in ["classification", "multilabel"] else train_segmentation_transform
    eval_transform = eval_classification_transform if args.task_type in ["classification", "multilabel"] else eval_segmentation_transform

    train_collate_fn = partial(multimodal_collate_fn, transform=train_transform, normalization=standard_transform)
    eval_collate_fn = partial(multimodal_collate_fn, transform=eval_transform, normalization=standard_transform)

    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=args.train_batch_size,
        collate_fn=train_collate_fn,
        num_workers = args.dataloader_num_workers,
        pin_memory = args.dataloader_pin_memory,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        dataset['val'],
        batch_size=args.eval_batch_size,
        collate_fn=eval_collate_fn,
        num_workers = args.dataloader_num_workers,
        pin_memory = args.dataloader_pin_memory,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        dataset['test'],
        batch_size=args.eval_batch_size,
        collate_fn=eval_collate_fn,
        num_workers = args.dataloader_num_workers,
        pin_memory = args.dataloader_pin_memory,
        shuffle=False,
    )

    dataloader = {
        "train": train_dataloader,
        "val": eval_dataloader,
        "test": test_dataloader,
    }

    data_collator = DataCollator(
        collate_fn_train=train_collate_fn,
        collate_fn_eval=eval_collate_fn,
    )

    return dataset, dataloader, data_collator # dataloader may be redundant
