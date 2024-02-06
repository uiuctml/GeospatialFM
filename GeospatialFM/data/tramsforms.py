# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence
import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)
    
class MaybeToTensorALL(object):
    def __init__(self, ignored_keys=['label']):
        self.ignored_keys = ignored_keys

    def __call__(self, samples):
        for key, val in samples.items():
            if key not in self.ignored_keys:
                if isinstance(val, torch.Tensor):
                    samples[key] = val
                else:
                    samples[key] = F.to_tensor(val)
        return samples
    
class RandomHorizontalFlipALL(object):
    def __init__(self, p=0.5, ignored_keys=['label']):
        self.p = p
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        if random.random() > self.p:
            for key, val in samples.items():
                if key not in self.ignored_keys:
                    samples[key] = F.hflip(val)
        return samples

class RandomRotationALL(object):
    def __init__(self, degrees, ignored_keys=['label']):
        self.degrees = (-degrees, degrees)
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = F.rotate(val, angle)
        return samples

class RandomResizedCropALL(object):
    """Randomly resize and crop the given PIL Image and its mask to a given size."""
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True, ignored_keys=['label']):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        i, j, h, w = transforms.RandomResizedCrop.get_params(samples.values()[0], self.scale, self.ratio)
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = F.resized_crop(val, i, j, h, w, self.size, interpolation=self.interpolation, antialias=self.antialias)
        return samples
    
class ResizeALL(object):
    """Resize the input PIL Image to the given size."""
    
    def __init__(self, size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True, ignored_keys=['label']):
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = F.resize(val, self.size, interpolation=self.interpolation, antialias=self.antialias)
        return samples
    
class CenterCropALL(object):
    """Crops the given PIL Image at the center."""
    
    def __init__(self, size, ignored_keys=['label']):
        self.size = size
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = F.center_crop(val, self.size)
        return samples
    
class StandardizeALL(object):
    """Standardize the input PIL Image."""
    
    def __init__(self, ignored_keys=['label', 'mask']):
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = (val / 10000.0).to(torch.float)
        return samples

def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    hflip_prob: float = 0.5,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
    **kwargs
):
    transforms_list = [RandomResizedCropALL(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(RandomHorizontalFlipALL(p=hflip_prob))
    transforms_list.append(MaybeToTensorALL())
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        raise NotImplementedError
    return transforms.Compose(transforms_list)

def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    crop_size: int = 224,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
    **kwargs
) -> transforms.Compose:
    transforms_list = [
        ResizeALL(resize_size, interpolation=interpolation),
        CenterCropALL(crop_size),
        MaybeToTensorALL(),
    ]
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        raise NotImplementedError
    return transforms.Compose(transforms_list)

def make_segmentation_train_transform(
    *,
    resize_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    hflip_prob: float = 0.5,
    random_rotate: bool = True,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
    **kwargs
):
    transforms_list = [ResizeALL(resize_size, interpolation=interpolation, ignored_keys=['label', 'mask'])]
    if hflip_prob > 0.0:
        transforms_list.append(RandomHorizontalFlipALL(p=hflip_prob))
    if random_rotate:
        transforms_list.append(RandomRotationALL(degrees=30))
    transforms_list.append(MaybeToTensorALL())
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        raise NotImplementedError
    return transforms.Compose(transforms_list)

def make_segmentation_eval_transform(
    *,
    resize_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
    **kwargs
) -> transforms.Compose:
    transforms_list = [
        ResizeALL(resize_size, interpolation=interpolation, ignored_keys=['label', 'mask']),
        MaybeToTensorALL(),
    ]
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        raise NotImplementedError
    return transforms.Compose(transforms_list)


# def make_standardize_transform():
#     return lambda x: (x / 10000.0).to(torch.float)
#     # return lambda x: x

# class TransformSample():
#     def __init__(self, transform, sar_transform=None):
#         self.transform = transform
#         self.sar_transform = sar_transform

#     def transform_img(self, sample):
#         if 'image1' in sample:
#             # OSCD # FIXME: Incorrect!
#             image1 = sample['image1']
#             image1 = self.transform(image1)
#             sample['image1'] = image1
#             image2 = sample['image2']
#             image2 = self.transform(image2)
#             sample['image2'] = image2
#             return sample
             
#         img = sample['image']
#         img = self.transform(img)
#         sample['image'] = img

#         if 'radar' in sample:
#             assert self.sar_transform is not None
#             radar = sample['radar']
#             radar = self.sar_transform(radar)
#             sample['radar'] = radar
#         return sample

#     def __call__(self, sample):
#         return self.transform_img(sample)


# # This roughly matches torchvision's preset for classification training:
# #   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
# def make_classification_train_transform(
#     *,
#     crop_size: int = 224,
#     interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
#     hflip_prob: float = 0.5,
#     mean_std: tuple = (None, None),
#     normalize: bool = False,
#     standardize: bool = True,
# ):
#     transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True)]
#     if hflip_prob > 0.0:
#         transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
#     transforms_list.append(MaybeToTensor())
#     if mean_std[0] is not None and normalize:
#         mean, std = mean_std
#         transforms_list.append(transforms.Normalize(mean, std))
#     elif standardize:
#         transforms_list.append(make_standardize_transform())
    
#     return transforms.Compose(transforms_list)


# # This matches (roughly) torchvision's preset for classification evaluation:
# #   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
# def make_classification_eval_transform(
#     *,
#     resize_size: int = 256,
#     interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
#     crop_size: int = 224,
#     mean_std: tuple = (None, None),
#     normalize: bool = False,
#     standardize: bool = True,
# ) -> transforms.Compose:
#     transforms_list = [
#         transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
#         transforms.CenterCrop(crop_size),
#         MaybeToTensor(),
#     ]
#     if mean_std[0] is not None and normalize:
#         mean, std = mean_std
#         transforms_list.append(transforms.Normalize(mean, std))
#     elif standardize:
#         transforms_list.append(make_standardize_transform())
    
#     return transforms.Compose(transforms_list)
