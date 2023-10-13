# from .utils import *

# Torchgeo is not compatible with the latest timm.
# from transformers import ViTModel, ViTConfig
# import timm
from typing import List, Optional, Union
import torchgeo.models as tgm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import segmentation_models_pytorch as smp

def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(classifier.cpu(), f)
    torch.save(classifier, save_path,)

def torch_load(save_path, device=None):
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)
    classifier = torch.load(save_path)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

class EncoderDecoder(nn.Module):
    def __init__(self, base_model, task_head, criterion=None, freeze_encoder=False):
        super().__init__()
        self.base_model = base_model
        self.task_head = task_head
        self.freeze_encoder = freeze_encoder
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # if freeze_encoder:
        #     # base_model requires grad = False
        #     for param in self.base_model.parameters():
        #         param.requires_grad = False
        #     self.base_model.eval()

    def forward(self, image, labels=None):
        if self.freeze_encoder:
            with torch.no_grad():
                feature = self.base_model(image)
        else:
            feature = self.base_model(image)
        logits = self.task_head(feature)
    
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        return logits
    
    def encode(self, x):
        feature = self.base_model(x)
        return feature
    
    def predict(self, x):
        logits = self.task_head(x)
        pred = torch.argmax(logits, dim=1)
        return pred
    
    def save(self, filename, verbose=True):
        if verbose: print(f'Saving Model to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading Model from {filename}')
        return torch_load(filename)

    
class ClassificationHead(nn.Linear):
    def __init__(self, out_features, in_features, weights=None, biases=None, use_bias=True, **kwargs):
        if weights is not None:
            out_features, in_features = weights.shape
        super().__init__(in_features, out_features, bias=use_bias)

        if weights is not None:
            self.weight = nn.Parameter(weights.clone())

        if use_bias:
            if biases is not None:
                self.bias = nn.Parameter(biases.clone())
            else:
                self.bias = nn.Parameter(torch.zeros_like(self.bias))
        self.weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, x):
        logits = F.linear(x, self.weight, self.bias)
        return logits

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger is not None:
            logger.info(f'Loading classification head from {filename}')
        return torch_load(filename)


class Unet(smp.Unet):
    def __init__(self, criterion=None, freeze_encoder=False, **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.freeze_encoder = freeze_encoder

    def forward(self, image, labels=None):
        """Sequentially pass `image` trough model`s encoder, decoder and heads"""

        self.check_input_shape(image)

        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(image)
        else:  
            features = self.encoder(image)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        if labels is not None:
            loss = self.criterion(masks, labels)
            return loss, masks

        return masks

class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_bathcnorm=use_bathcnorm,
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        use_batchnorm=True,
        out_channels=512,
        dropout=0.2,
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = modules.Conv2dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x