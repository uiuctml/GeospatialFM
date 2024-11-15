import torch
import torchvision.transforms.functional as TF
import numpy as np
from functools import partial
from .transforms import ResizeAll

# collate function for dataloader of multimodal data
def multimodal_collate_fn(batch, transform=None, random_crop=False, scale=None, crop_size=None):
    optical_list, radar_list = [], []
    optical_channel_wv, radar_channel_wv = None, None
    spatial_resolution = None
    
    if random_crop:
        crop_scale = np.random.choice([1, 2])
        crop_size = crop_size // crop_scale
        scale = scale * crop_scale

    for example in batch:
        # to tensor
        example['optical'] = torch.tensor(example['optical'])
        example['radar'] = torch.tensor(example['radar'])
        example['optical_channel_wv'] = torch.tensor(example['optical_channel_wv']).unsqueeze(0)
        example['radar_channel_wv'] = torch.tensor(example['radar_channel_wv']).unsqueeze(0)
        example['spatial_resolution'] = example['spatial_resolution']
        
        if transform is not None:
            example = transform(example, crop_size=crop_size, scale=scale)
            
        if optical_channel_wv is None and radar_channel_wv is None:
            optical_channel_wv = example['optical_channel_wv']
            radar_channel_wv = example['radar_channel_wv']
        else:
            # ensure the same optical and radar channel wv across the batch
            assert (example['optical_channel_wv'] == optical_channel_wv).all() 
            assert (example['radar_channel_wv'] == radar_channel_wv).all()

        if spatial_resolution is None:
            spatial_resolution = example['spatial_resolution']
        else:
            assert example['spatial_resolution'] == spatial_resolution
            
        optical_list.append(example['optical'])
        radar_list.append(example['radar'])
    
    assert optical_channel_wv is not None and radar_channel_wv is not None
    assert spatial_resolution is not None
    
    return {
        'optical': torch.stack(optical_list),
        'radar': torch.stack(radar_list),
        'optical_channel_wv': optical_channel_wv,
        'radar_channel_wv': radar_channel_wv,
        'spatial_resolution': spatial_resolution
    }
