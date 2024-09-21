import torch
import torchvision.transforms.functional as TF
import numpy as np

# data normalization applied to datasets directly
def apply_transforms(example, optical_mean, optical_std, radar_mean, radar_std, use_8bit=False):
    # Apply your transforms here
    optical = example.get('optical')
    radar = example.get('radar')
        
    # normalize
    def normalize(x, mean, std):
        x = x.float()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        min_values = torch.tensor(mean) - 2 * torch.tensor(std)
        max_values = torch.tensor(mean) + 2 * torch.tensor(std)            
        
        x_normalized = (x - min_values[None, :, None, None]) / (max_values[None, :, None, None] - min_values[None, :, None, None])
        
        if use_8bit:
            x_clipped = x_normalized * 255
            x_clipped = torch.clip(x_clipped, 0, 255).to(torch.uint8)
        else:
            x_clipped = torch.clip(x_normalized, 0, 1)
            
        return x_clipped.squeeze(0)
    
    if optical is not None:
        # to tensor
        optical = torch.tensor(optical)
        # center crop
        optical = TF.center_crop(optical, [256, 256])
        # normalize
        optical = normalize(optical, optical_mean, optical_std)
        example['optical'] = optical.numpy()
        
    if radar is not None:
        radar = torch.tensor(radar)
        # center crop
        radar = TF.center_crop(radar, [256, 256])
        # normalize
        radar = normalize(radar, radar_mean, radar_std)
        example['radar'] = radar.numpy()
    
    return example

# collate function for dataloader of multimodal data
def multimodal_collate_fn(batch, transform=None, random_crop=True, normalization=None):
    optical_list, radar_list = [], []
    optical_channel_wv, radar_channel_wv = None, None
    spatial_resolution = None
    
    crop_size = np.random.choice([128, 224, 256]) if random_crop else None

    for example in batch:
        if normalization:
            example = normalization(example)
        # to tensor
        example['optical'] = torch.tensor(example['optical'])
        example['radar'] = torch.tensor(example['radar'])
        example['optical_channel_wv'] = torch.tensor(example['optical_channel_wv']).unsqueeze(0)
        example['radar_channel_wv'] = torch.tensor(example['radar_channel_wv']).unsqueeze(0)
        example['spatial_resolution'] = example['spatial_resolution']
        
        if transform:
            example = transform(example, crop_size)
            
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
