import os
import torch
import torchgeo.models as tgm
import torch.nn as nn
import timm
from .vision_transformer import ViTEncoder, ViTDecoder
from .mae import CrossModalMAEViT
from collections import OrderedDict
import numpy as np

class ViTModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        x = self.encoder(x, return_dict=True)['cls_token']
        x = self.head(x)
        return x
    
def unwrap_model(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
    return new_state_dict

def decompose_model(state_dict):
    optical_model = OrderedDict()
    radar_model = OrderedDict()
    for key, value in state_dict.items():
        try:
            key_prefix, key_name = key.split('.', 1)
        except:
            print(key)
            continue
        if 'optical_encoder' in key_prefix:
            optical_model[key_name] = value
        elif 'radar_encoder' in key_prefix:
            radar_model[key_name] = value
    return optical_model, radar_model

def construct_encoder(model_cfg, arch=None):
    assert model_cfg['load_pretrained_from'] in ['timm', 'torchgeo', 'dir', None]
    if arch is None: assert model_cfg['load_pretrained_from'] in ['dir', None]

    encoder = ViTEncoder(**model_cfg['kwargs'])

    if model_cfg['load_pretrained_from'] != None:
        if model_cfg['load_pretrained_from'] == 'torchgeo':
            assert model_cfg['pretrained_ckpt'] is not None
            weights = tgm.get_weight(model_cfg['pretrained_ckpt'])
            state_dict = tgm.get_model(arch, weights=weights).state_dict()

        elif model_cfg['load_pretrained_from'] == 'timm':
            state_dict = timm.create_model(arch, pretrained=(model_cfg['load_pretrained_from']=='timm')).state_dict()  

        elif model_cfg['load_pretrained_from'] == 'dir':
            state_dict = torch.load(model_cfg['pretrained_ckpt'])

        try:
            encoder.load_state_dict(state_dict, strict=False)
        except:
            raise ValueError(f"Pretrained ckpt for {arch} mismatch with model config")

    if model_cfg['freeze_encoder']:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def construct_decoder(decoder_cfg):
    return ViTDecoder(**decoder_cfg['kwargs'])

def construct_mae(model_cfg):
    arch = model_cfg['architecture']
    if model_cfg['cross_modal']:
        optical_encoder = construct_encoder(model_cfg['OPTICAL'], arch=arch)
        radar_encoder = construct_encoder(model_cfg['RADAR'], arch=arch)

        if model_cfg['unified_decoder']:
            decoder = construct_decoder(model_cfg['DECODER'])
            optical_decoder = radar_decoder = decoder
        else:
            optical_decoder = construct_decoder(model_cfg['OPTICAL_DECODER'])
            radar_decoder = construct_decoder(model_cfg['RADAR_DECODER'])

        logit_scale = np.log(10) if model_cfg['use_siglip'] else np.log(1 / 0.07)
        logit_bias = -10 if model_cfg['use_siglip'] else None
        
        mae = CrossModalMAEViT(optical_encoder, radar_encoder, optical_decoder, radar_decoder, init_logit_scale=logit_scale, init_logit_bias=logit_bias, use_clip=model_cfg['use_clip'])
        if model_cfg['OPTICAL']['use_head'] and model_cfg['RADAR']['use_head']:
            optical_head = construct_head(model_cfg['OPTICAL']['head_kwargs'])
            radar_head = construct_head(model_cfg['RADAR']['head_kwargs'])
            mae.set_head(optical_head, radar_head)
    else:
        raise NotImplementedError
    return mae

def construct_downstream_models(model_cfg):
    modals = ['OPTICAL', 'RADAR']
    arch = model_cfg['architecture']
    models = {}
    for modal in modals:
        if not hasattr(model_cfg, modal): continue
        assert model_cfg[modal]['use_head']
        encoder = construct_encoder(model_cfg[modal], arch=arch)
        head = construct_head(model_cfg[modal]['head_kwargs'])
        model = ViTModel(encoder, head)
        models[modal] = model
    return models

def construct_head(head_cfg):
    if head_cfg['task_type'] == 'classification':
        head = nn.Linear(head_cfg['in_features'], head_cfg['num_classes'], bias=head_cfg['use_bias'])
    else:
        raise NotImplementedError
    return head


# def build_crop(cfg):
#     optical = construct_model(cfg['MODEL'])
#     optical_encoder = optical.base_model if hasattr(optical, 'base_model') else optical
#     sar = construct_model(cfg['SAR_MODEL'])
#     sar_encoder = sar.base_model if hasattr(sar, 'base_model') else sar
#     crop = CROP().align_pretrained(optical_encoder, sar_encoder)
#     return crop