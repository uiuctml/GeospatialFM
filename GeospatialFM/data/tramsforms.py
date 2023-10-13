# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms


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


def make_standardize_transform():
    return lambda x: (x / 10000.0).to(torch.float)
    # return lambda x: x

class TransformSample():
    def __init__(self, transform,):
        self.transform = transform

    def transform_img(self, sample):
        img = sample['image']
        img = self.transform(img)
        sample['image'] = img
        return sample

    def __call__(self, sample):
        return self.transform_img(sample)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    hflip_prob: float = 0.5,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(MaybeToTensor())
    if mean_std[0] is not None and normalize:
        mean, std = mean_std
        transforms_list.append(transforms.Normalize(mean, std))
    elif standardize:
        transforms_list.append(make_standardize_transform())
    
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC, # BILINEAR
    crop_size: int = 224,
    mean_std: tuple = (None, None),
    normalize: bool = False,
    standardize: bool = True,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
    ]
    if mean_std[0] is not None and normalize:
        mean, std = mean_std
        transforms_list.append(transforms.Normalize(mean, std))
    elif standardize:
        transforms_list.append(make_standardize_transform())
    
    return transforms.Compose(transforms_list)
