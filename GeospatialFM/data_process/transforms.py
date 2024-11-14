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
    

def pretrain_transform(example, crop_size=None, scale=None):
    optical = example['optical']
    radar = example['radar']
    
    # random crop
    if crop_size is not None:
        i, j, h, w = transforms.RandomCrop.get_params(optical, [crop_size, crop_size])
        optical = TF.crop(optical, i, j, h, w)
        radar = TF.crop(radar, i, j, h, w)
    
    # horizontal flip
    if np.random.random() < 0.5:
        optical = TF.hflip(optical)
        radar = TF.hflip(radar)
    
    # vertical flip
    if np.random.random() < 0.5:
        optical = TF.vflip(optical)
        radar = TF.vflip(radar)
        
    if scale is not None:
        optical = TF.resize(optical, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        radar = TF.resize(radar, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        example['spatial_resolution'] = example['spatial_resolution'] / scale
    
    example['optical'] = optical
    example['radar'] = radar

    return example

def segmentation_transform_one_sample(optical, radar, label, spatial_resolution, crop_size=None, scale=None, is_train=True, random_rotation=True, 
                                       optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
    # Convert lists directly to tensors
    optical = None if optical is None else torch.tensor(optical, dtype=torch.float32)
    radar = None if radar is None else torch.tensor(radar, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int64).unsqueeze(0)
    
    # normalize
    optical, radar = NormalizeAll(optical, radar, optical_mean, optical_std, radar_mean, radar_std)

    # random crop
    if crop_size is not None and is_train:
        optical, radar, label = RandomCropAll(optical, radar, label, crop_size)
    elif crop_size is not None and not is_train:
        optical, radar, label = CenterCropAll(optical, radar, label, crop_size)
    
    # Train-time augmentations
    if is_train:
        # horizontal flip
        if np.random.random() < 0.5:
            optical, radar, label = HorizontalFlipAll(optical, radar, label)    
        # vertical flip
        if np.random.random() < 0.5:
            optical, radar, label = VerticalFlipAll(optical, radar, label)
        # random rotation 90
        if random_rotation:
            optical, radar, label = RandomRotationAll(optical, radar, label)
    
    if scale is not None:
        optical, radar = ResizeAll(optical, radar, scale, crop_size)
        spatial_resolution = spatial_resolution / scale
    label = label.squeeze(0)
    
    return optical, radar, label, spatial_resolution

def segmentation_transform(example, crop_size=None, scale=None, is_train=True, random_rotation=True, 
                           optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
    optical = example.get('optical', None)
    radar = example.get('radar', None)
    label = example.get('label', None)
    spatial_resolution = example.get('spatial_resolution', None)
    assert label is not None
    
    optical_list = []
    radar_list = []
    label_list = []
    spatial_resolution_list = []
    
    if optical is not None:
        num_samples = len(optical)
    elif radar is not None:
        num_samples = len(radar)
    else:
        num_samples = len(label)
    
    for i in range(num_samples):
        optical_i = None if optical is None else optical[i]
        radar_i = None if radar is None else radar[i]
        label_i = label[i]
        spatial_resolution_i = spatial_resolution[i] if spatial_resolution is not None else None
        
        optical_i, radar_i, label_i, spatial_resolution_i = segmentation_transform_one_sample(
            optical_i, radar_i, label_i, spatial_resolution_i, 
            crop_size, scale, is_train, random_rotation, optical_mean,
            optical_std, radar_mean, radar_std
        )

        if optical_i is not None:
            optical_list.append(optical_i)
        if radar_i is not None:
            radar_list.append(radar_i)
        label_list.append(label_i)
        if spatial_resolution_i is not None:
            spatial_resolution_list.append(spatial_resolution_i)
    
    if optical_list:
        example['optical'] = optical_list
        # example['optical_channel_wv'] = torch.tensor(example['optical_channel_wv'])
    if radar_list:
        example['radar'] = radar_list
        # example['radar_channel_wv'] = torch.tensor(example['radar_channel_wv'])
    example['label'] = label_list
    if spatial_resolution_list:
        example['spatial_resolution'] = spatial_resolution_list
    
    return example

def classification_transform_one_sample(optical, radar, spatial_resolution, crop_size=None, scale=None, is_train=True, random_rotation=True, 
                                        optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
    # Convert lists directly to tensors
    optical = None if optical is None else torch.tensor(optical, dtype=torch.float32)
    radar = None if radar is None else torch.tensor(radar, dtype=torch.float32)
    
    # normalize
    optical, radar = NormalizeAll(optical, radar, optical_mean, optical_std, radar_mean, radar_std)
    
    # random crop
    if crop_size is not None and is_train:
        optical, radar, _ = RandomCropAll(optical, radar, crop_size=crop_size)
    elif crop_size is not None and not is_train:
        optical, radar, _ = CenterCropAll(optical, radar, crop_size=crop_size)
    
    # Train-time augmentations
    if is_train:
        # horizontal flip
        if np.random.random() < 0.5:
            optical, radar, _ = HorizontalFlipAll(optical, radar)
        # vertical flip
        if np.random.random() < 0.5:
            optical, radar, _ = VerticalFlipAll(optical, radar)
        # random rotation 90
        if random_rotation:
            optical, radar, _ = RandomRotationAll(optical, radar)
    
    if scale is not None:
        optical, radar = ResizeAll(optical, radar, scale, crop_size)
        spatial_resolution = spatial_resolution / scale
        
    return optical, radar, spatial_resolution

def classification_transform(example, crop_size=None, scale=None, is_train=True, random_rotation=True, 
                             optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
    optical = example.get('optical', None)
    radar = example.get('radar', None)
    spatial_resolution = example.get('spatial_resolution', None)
    optical_list = []
    radar_list = []
    spatial_resolution_list = []
    
    if optical is not None:
        num_samples = len(optical)
    else:
        num_samples = len(radar)
    
    for i in range(num_samples):
        optical_i = None if optical is None else optical[i]
        radar_i = None if radar is None else radar[i]
        spatial_resolution_i = spatial_resolution[i] if spatial_resolution is not None else None
        optical_i, radar_i, spatial_resolution_i = classification_transform_one_sample(optical_i, radar_i, spatial_resolution_i, crop_size, scale, is_train, random_rotation,
                                                                                       optical_mean, optical_std, radar_mean, radar_std)
        if optical_i is not None:
            optical_list.append(optical_i)
        if radar_i is not None:
            radar_list.append(radar_i)
        if spatial_resolution_i is not None:
            spatial_resolution_list.append(spatial_resolution_i)
    
    if optical_list:
        example['optical'] = optical_list
    if radar_list:
        example['radar'] = radar_list
    if spatial_resolution_list:
        example['spatial_resolution'] = spatial_resolution_list
    
    return example

def get_transform(task_type, crop_size=None, scale=None, random_rotation=True, optical_mean=None, optical_std=None, radar_mean=None, radar_std=None):
    if task_type == "segmentation":
        train_transform = partial(segmentation_transform, crop_size=crop_size, scale=scale, random_rotation=random_rotation, is_train=True, 
                                  optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std)
        eval_transform = partial(segmentation_transform, crop_size=crop_size, scale=scale, is_train=False, 
                                  optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std)
    elif task_type == "classification" or task_type == "multilabel":
        train_transform = partial(classification_transform, crop_size=crop_size, scale=scale, random_rotation=random_rotation, is_train=True, 
                                  optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std)
        eval_transform = partial(classification_transform, crop_size=crop_size, scale=scale, is_train=False, 
                                  optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std)
    else:
        raise NotImplementedError
    
    return train_transform, eval_transform