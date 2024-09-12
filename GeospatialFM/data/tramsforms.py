import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF

def pretrain_transform(example, crop_size=None):
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
    
    example['optical'] = optical
    example['radar'] = radar

    return example
