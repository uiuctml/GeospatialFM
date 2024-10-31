import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from functools import partial

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

def train_segmentation_transform(example, crop_size=None, scale=None, random_rotation=True):
    optical = example.get('optical', None)
    radar = example.get('radar', None)
    label = example.get('label', None)
    
    # random crop
    if crop_size is not None:
        i, j, h, w = transforms.RandomCrop.get_params(optical, [crop_size, crop_size])
        optical = TF.crop(optical, i, j, h, w) if optical is not None else None
        radar = TF.crop(radar, i, j, h, w) if radar is not None else None
        label = TF.crop(label, i, j, h, w) if label is not None else None
    
    # horizontal flip
    if np.random.random() < 0.5:
        optical = TF.hflip(optical) if optical is not None else None
        radar = TF.hflip(radar) if radar is not None else None
        label = TF.hflip(label) if label is not None else None
        
    # vertical flip
    if np.random.random() < 0.5:
        optical = TF.vflip(optical) if optical is not None else None
        radar = TF.vflip(radar) if radar is not None else None
        label = TF.vflip(label) if label is not None else None

    # random rotation 90
    if random_rotation:
        k = np.random.randint(0, 4)  # 0-3 for number of 90-degree rotations
        optical = TF.rotate(optical, angle=90*k) if optical is not None else None
        radar = TF.rotate(radar, angle=90*k) if radar is not None else None
        label = TF.rotate(label, angle=90*k) if label is not None else None
    
    if scale is not None:
        optical = TF.resize(optical, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if optical is not None else None
        radar = TF.resize(radar, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if radar is not None else None
        example['spatial_resolution'] = example['spatial_resolution'] / scale
    
    example['optical'] = optical
    example['radar'] = radar
    example['label'] = label

    return example

def eval_segmentation_transform(example, crop_size=None, scale=None, ):
    optical = example.get('optical', None)
    radar = example.get('radar', None)
    label = example.get('label', None)
    
    # center crop
    if crop_size is not None:
        optical = TF.center_crop(optical, crop_size) if optical is not None else None
        radar = TF.center_crop(radar, crop_size) if radar is not None else None
        label = TF.center_crop(label, crop_size) if label is not None else None
    
    if scale is not None:
        optical = TF.resize(optical, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if optical is not None else None
        radar = TF.resize(radar, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if radar is not None else None
        example['spatial_resolution'] = example['spatial_resolution'] / scale
    
    example['optical'] = optical
    example['radar'] = radar
    example['label'] = label

    return example

def train_classification_transform(example, crop_size=None, scale=None, random_rotation=True):
    optical = example.get('optical', None)
    radar = example.get('radar', None)
    
    # random crop
    if crop_size is not None:
        i, j, h, w = transforms.RandomCrop.get_params(optical, [crop_size, crop_size])
        optical = TF.crop(optical, i, j, h, w) if optical is not None else None
        radar = TF.crop(radar, i, j, h, w) if radar is not None else None
    
    # horizontal flip
    if np.random.random() < 0.5:
        optical = TF.hflip(optical) if optical is not None else None
        radar = TF.hflip(radar) if radar is not None else None
        
    # vertical flip
    if np.random.random() < 0.5:
        optical = TF.vflip(optical) if optical is not None else None
        radar = TF.vflip(radar) if radar is not None else None

    # random rotation 90
    if random_rotation:
        k = np.random.randint(0, 4)  # 0-3 for number of 90-degree rotations
        optical = TF.rotate(optical, angle=90*k) if optical is not None else None
        radar = TF.rotate(radar, angle=90*k) if radar is not None else None
    
    if scale is not None:
        optical = TF.resize(optical, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if optical is not None else None
        radar = TF.resize(radar, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if radar is not None else None
        example['spatial_resolution'] = example['spatial_resolution'] / scale
    
    example['optical'] = optical
    example['radar'] = radar

    return example

def eval_classification_transform(example, crop_size=None, scale=None):
    optical = example.get('optical', None)
    radar = example.get('radar', None)
    
    # center crop
    if crop_size is not None:
        optical = TF.center_crop(optical, crop_size) if optical is not None else None
        radar = TF.center_crop(radar, crop_size) if radar is not None else None
    
    if scale is not None:
        optical = TF.resize(optical, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if optical is not None else None
        radar = TF.resize(radar, scale*crop_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True) if radar is not None else None
        example['spatial_resolution'] = example['spatial_resolution'] / scale
    
    example['optical'] = optical
    example['radar'] = radar

    return example

def get_transform(task_type, crop_size=None, scale=None, random_rotation=True):
    if task_type == "segmentation":
        train_transform = partial(train_segmentation_transform, crop_size=crop_size, scale=scale, random_rotation=random_rotation)
        eval_transform = partial(eval_segmentation_transform, crop_size=crop_size, scale=scale)
    elif task_type == "classification" or task_type == "multilabel":
        train_transform = partial(train_classification_transform, crop_size=crop_size, scale=scale, random_rotation=random_rotation)
        eval_transform = partial(eval_classification_transform, crop_size=crop_size, scale=scale)
    else:
        raise NotImplementedError
    
    return train_transform, eval_transform