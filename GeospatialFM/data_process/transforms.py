import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from functools import partial
import torch

def NormalizeAll(optical=None, radar=None, optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
     # normalize
    def normalize(x, mean, std):
        x = x.float()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        min_values = torch.tensor(mean) - 2 * torch.tensor(std)
        max_values = torch.tensor(mean) + 2 * torch.tensor(std)            
        
        x_normalized = (x - min_values[None, :, None, None]) / (max_values[None, :, None, None] - min_values[None, :, None, None])
        x_clipped = torch.clip(x_normalized, 0, 1)
            
        return x_clipped.squeeze(0)
    
    if optical is not None:
        assert optical_mean is not None and optical_std is not None
        # to tensor
        if not isinstance(optical, torch.Tensor):
            optical = torch.tensor(optical)
        optical = normalize(optical, optical_mean, optical_std)
        
    if radar is not None:
        assert radar_mean is not None and radar_std is not None
        # to tensor
        if not isinstance(radar, torch.Tensor):
            radar = torch.tensor(radar)
        radar = normalize(radar, radar_mean, radar_std)
    
    return optical, radar

def RandomCropAll(optical=None, radar=None, label=None, crop_size=None):
    i, j, h, w = transforms.RandomCrop.get_params(optical, [crop_size, crop_size])
    optical = ModuleNotFoundError if optical is None else TF.crop(optical, i, j, h, w)
    radar = None if radar is None else TF.crop(radar, i, j, h, w)
    label = None if label is None else TF.crop(label, i, j, h, w)
    return optical, radar, label

def CenterCropAll(optical=None, radar=None, label=None, crop_size=None):
    optical = None if optical is None else TF.center_crop(optical, crop_size)
    radar = None if radar is None else TF.center_crop(radar, crop_size)
    label = None if label is None else TF.center_crop(label, crop_size)
    return optical, radar, label

def HorizontalFlipAll(optical=None, radar=None, label=None):
    optical = None if optical is None else TF.hflip(optical)
    radar = None if radar is None else TF.hflip(radar)
    label = None if label is None else TF.hflip(label)
    return optical, radar, label

def VerticalFlipAll(optical=None, radar=None, label=None):
    optical = None if optical is None else TF.vflip(optical)
    radar = None if radar is None else TF.vflip(radar)
    label = None if label is None else TF.vflip(label)
    return optical, radar, label

def RandomRotationAll(optical=None, radar=None, label=None):
    k = np.random.randint(0, 4)  # 0-3 for number of 90-degree rotations
    optical = None if optical is None else TF.rotate(optical, 90*k)
    radar = None if radar is None else TF.rotate(radar, 90*k)
    label = None if label is None else TF.rotate(label, 90*k)
    return optical, radar, label

def ResizeAll(optical=None, radar=None, scale=None, crop_size=None):
    optical = None if optical is None else TF.resize(optical, int(scale*crop_size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    radar = None if radar is None else TF.resize(radar, int(scale*crop_size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    return optical, radar

def pretrain_transform(example, crop_size=None, scale=None, optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
    optical = example['optical']
    radar = example['radar']

    # normalization
    optical, radar = NormalizeAll(optical, radar, optical_mean, optical_std, radar_mean, radar_std)
    
    # random crop
    if crop_size is not None:
        optical, radar, _ = RandomCropAll(optical, radar, None, crop_size)
    
    # horizontal flip
    if np.random.random() < 0.5:
        optical, radar, _ = HorizontalFlipAll(optical, radar, None)
    
    # vertical flip
    if np.random.random() < 0.5:
        optical, radar, _ = VerticalFlipAll(optical, radar, None)

    # resize
    if scale is not None:
        optical, radar = ResizeAll(optical, radar, scale, crop_size)
        example['spatial_resolution'] = example['spatial_resolution'] / scale
    
    example['optical'] = optical
    example['radar'] = radar

    return example
