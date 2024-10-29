import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF

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

def train_classification_transform(): # TODO
    pass

def eval_classification_transform(): # TODO
    pass

def train_segmentation_transform(): # TODO
    pass

def eval_segmentation_transform(): # TODO
    pass
