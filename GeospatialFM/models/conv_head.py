import os
from typing import Any, Optional, Dict
import timm
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# Temporarily import wandb and matplotlib to log images
import wandb
import matplotlib.pyplot as plt
import warnings
from typing import List


class ConvHead(nn.Module):
    def __init__(self, embedding_size: int = 384, num_classes: int = 5, patch_size: int = 4):
        super(ConvHead, self).__init__()

        # Ensure patch_size is a positive power of 2
        if not (patch_size > 0 and ((patch_size & (patch_size - 1)) == 0)):
            raise ValueError("patch_size must be a positive power of 2.")

        num_upsampling_steps = int(math.log2(patch_size))

        # Determine the initial number of filters (maximum 128 or embedding_size)
        initial_filters = 128

        # Generate the sequence of filters: 128, 64, 32, ..., down to num_classes
        filters = [initial_filters // (2 ** i) for i in range(num_upsampling_steps - 1)]
        filters.append(num_classes)  # Ensure the last layer outputs num_classes channels

        layers = []
        in_channels = embedding_size

        for i in range(num_upsampling_steps):
            out_channels = filters[i]

            # Upsampling layer
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

            # Convolutional layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            # Apply BatchNorm and ReLU only if not the last layer
            if i < num_upsampling_steps - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels  # Update in_channels for the next iteration

        self.segmentation_conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.segmentation_conv(x)