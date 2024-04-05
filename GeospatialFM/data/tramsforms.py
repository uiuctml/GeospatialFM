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

class RandomVerticalFlipALL(object):
    def __init__(self, p=0.5, ignored_keys=['label']):
        self.p = p
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        if random.random() > self.p:
            for key, val in samples.items():
                if key not in self.ignored_keys:
                    samples[key] = F.vflip(val)
        return samples


class RandomRotationALL(object):
    def __init__(self, degrees, ignored_keys=['label']):
        # self.degrees = (-degrees, degrees)
        self.degrees = [0, 90, 180, 270]
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        # angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        angle = random.choice(self.degrees)
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
        if 'image' in samples: image = samples['image']
        elif 'image1' in samples: image = samples['image1']
        elif 'radar' in samples: image = samples['radar']
        else: raise ValueError('No image found in samples')
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = F.resized_crop(val, i, j, h, w, [self.size, self.size], interpolation=self.interpolation, antialias=self.antialias)
        return samples
    
class RandomCropALL(object):
    """Randomly resize and crop the given PIL Image and its mask to a given size."""
    
    def __init__(self, size, ignored_keys=['label']):
        self.size = size
        self.ignored_keys = ignored_keys
    
    def __call__(self, samples):
        if 'image' in samples: image = samples['image']
        elif 'image1' in samples: image = samples['image1']
        elif 'radar' in samples: image = samples['radar']
        else: raise ValueError('No image found in samples')
        i, j, h, w = transforms.RandomCrop.get_params(image, [self.size, self.size])
        for key, val in samples.items():
            if key not in self.ignored_keys:
                samples[key] = F.crop(val, i, j, h, w)
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
                if key == 'radar':
                    samples[key] = (val / -1000.0).to(torch.float)
                else:
                    samples[key] = (val / 10000.0).to(torch.float)
        return samples
    
class NormalizeALL(object):
    """Normalize the input PIL Image."""
    
    def __init__(self, mean, std, ignored_keys=['label', 'mask']):
        self.mean = torch.tensor(mean) if not isinstance(mean, torch.Tensor) else mean
        self.std = torch.tensor(std) if not isinstance(std, torch.Tensor) else std
        self.ignored_keys = ignored_keys
        
    def _normalize(self, img, mean, std):
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        min_value = mean - 4 * std
        max_value = mean + 4 * std
        img = (img - min_value) / (max_value - min_value) * 255.0
        # img = torch.clip(img, 0, 1)
        img = torch.clip(img, 0, 255).to(torch.uint8)
        return img
    
    def __call__(self, samples):
        img = []
        split_point = 0
        if 'radar' in samples.keys() and 'radar' not in self.ignored_keys:
            img.append(samples['radar'])
            split_point = samples['radar'].shape[0]
        if 'image' in samples.keys() and 'image' not in self.ignored_keys:
            img.append(samples['image'])
        
        if len(img) > 0:
            img = torch.cat(img, dim=0)
            if img.shape[0] < len(self.mean) and split_point == 0:
                mean = self.mean[-img.shape[0]:]
                std = self.std[-img.shape[0]:]
            elif img.shape[0] < len(self.mean) and split_point > 0:
                mean = self.mean[:split_point]
                std = self.std[:split_point]
            else:
                mean = self.mean
                std = self.std
            assert img.shape[0] == len(mean) == len(std)
            img = F.normalize(img, mean, std)
            # img = self._normalize(img, mean, std)
            if 'radar' in samples.keys() and 'radar' not in self.ignored_keys:
                samples['radar'] = img[:split_point].float()
            if 'image' in samples.keys() and 'image' not in self.ignored_keys:
                samples['image'] = img[split_point:].float()
        else:
            assert 'image1' in samples.keys() and 'image2' in samples.keys()
            samples['image1'] = F.normalize(samples['image1'], self.mean, self.std)#.float()
            samples['image2'] = F.normalize(samples['image2'], self.mean, self.std)#.float()
            # samples['image1'] = self._normalize(samples['image1'], self.mean, self.std)
            # samples['image2'] = self._normalize(samples['image2'], self.mean, self.std)

        return samples


def make_pretrain_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    hflip_prob: float = 0.5,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
    **kwargs
):
    # transforms_list = [RandomResizedCropALL(crop_size, interpolation=interpolation, scale=(0.05, 0.4))]
    transforms_list = [RandomCropALL(120), RandomResizedCropALL(crop_size, interpolation=interpolation, scale=(0.08, 1))]
    # transforms_list = [CenterCropALL(crop_size)]
    if hflip_prob > 0.0:
        transforms_list.append(RandomHorizontalFlipALL(p=hflip_prob))
    transforms_list.append(MaybeToTensorALL())
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        transforms_list.append(NormalizeALL(mean_std[0], mean_std[1]))
    return transforms.Compose(transforms_list)

def make_eval_transform(
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
        transforms_list.append(NormalizeALL(mean_std[0], mean_std[1]))
    return transforms.Compose(transforms_list)

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
    transforms_list = [RandomResizedCropALL(crop_size, interpolation=interpolation, scale=(0.8, 1.0))]
    if hflip_prob > 0.0:
        transforms_list.append(RandomHorizontalFlipALL(p=hflip_prob))
    transforms_list.append(MaybeToTensorALL())
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        transforms_list.append(NormalizeALL(mean_std[0], mean_std[1]))
    return transforms.Compose(transforms_list)


def make_segmentation_train_transform(
    *,
    resize_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    random_rotate: bool = True,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
    **kwargs
):
    transforms_list = [ResizeALL(resize_size, interpolation=interpolation, ignored_keys=['label', 'mask'])]
    if hflip_prob > 0.0:
        transforms_list.append(RandomHorizontalFlipALL(p=hflip_prob))
    if vflip_prob > 0.0:
        transforms_list.append(RandomVerticalFlipALL(p=vflip_prob))
    if random_rotate:
        transforms_list.append(RandomRotationALL(degrees=90))
    transforms_list.append(MaybeToTensorALL())
    if standardize:
        transforms_list.append(StandardizeALL())
    elif normalize:
        transforms_list.append(NormalizeALL(mean_std[0], mean_std[1]))
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
        transforms_list.append(NormalizeALL(mean_std[0], mean_std[1]))
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
