from .utils import *

# Torchgeo is not compatible with the latest timm.
# from transformers import ViTModel, ViTConfig
# import timm
import torchgeo.models as tgm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ViTEncoderDecoder(nn.Module):
    def __init__(self, base_model, task_head, criterion=None, lp=False):
        super().__init__()
        self.base_model = base_model
        self.task_head = task_head
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        if lp:
            # base_model requires grad = False
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, image, labels=None):
        feature = self.base_model(image)
        logits = self.task_head(feature)
    
        if labels is not None:
            print(logits.shape, labels.shape)
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


class ViT(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        pass

        
