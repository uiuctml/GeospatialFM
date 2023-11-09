import os
import torch
import torchgeo.models as tgm
import torch.nn as nn
import timm
from .vision_transformer import ViTEncoder, ViTDecoder
from .mae import CrossModalMAEViT

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
        if model_cfg['OPTICAL']['use_head']:
            optical_head = construct_head(model_cfg['OPTICAL']['head_kwargs'])
        if model_cfg['RADAR']['use_head']:
            radar_head = construct_head(model_cfg['RADAR']['head_kwargs'])
    if model_cfg['cross_modal']:
        decoder = construct_decoder(model_cfg['DECODER'])
    mae = CrossModalMAEViT(optical_encoder, radar_encoder, decoder, decoder)
    if model_cfg['OPTICAL']['use_head'] and model_cfg['RADAR']['use_head']:
        mae.set_head(optical_head, radar_head)
    return mae

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