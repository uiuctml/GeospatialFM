import os
import torch
import torchgeo.models as tgm
import torch.nn as nn
import timm
from .vision_transformer import ViTEncoder, ViTDecoder
from .channel_attn_vit import ChannelViTEncoder
# from .channel_vit import ChannelViTEncoder
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

def get_pretrained_weight(cfg, arch, skip_modules=['patch_embed.proj.weight', 'patch_embed.proj.bias']):
    model_cfg = cfg.MODEL
    load_pretrained_from = model_cfg['load_pretrained_from']
    pretrained_ckpt = model_cfg['pretrained_ckpt'] if model_cfg['pretrained_ckpt'] is not None else 'final_model.pth'
    state_dicts = {}
    assert load_pretrained_from in ['timm', 'torchgeo', 'dir', None]
    if load_pretrained_from is None:
        return None
    elif load_pretrained_from == 'timm':
        print(f"Loading pretrained weights from {arch}.{pretrained_ckpt}...")
        state_dict = timm.create_model(f'{arch}.{pretrained_ckpt}', pretrained=True).state_dict()
        for skip_module in skip_modules:
            del state_dict[skip_module]
        state_dicts = {'OPTICAL': state_dict, 'RADAR': state_dict}
    elif load_pretrained_from == 'dir':
        save_path = os.path.join(cfg.TRAINER['ckpt_dir'], pretrained_ckpt)
        print(f"Loading pretrained weights from {save_path}...")
        state_dict = unwrap_model(torch.load(save_path, map_location='cpu'))
        optical_state_dict, radar_state_dict = decompose_model(state_dict)
        state_dicts = {'OPTICAL': optical_state_dict, 'RADAR': radar_state_dict}
    else:
        raise NotImplementedError
    return state_dicts

def construct_encoder(model_cfg, arch):
    # assert model_cfg['load_pretrained_from'] in ['timm', 'torchgeo', 'dir', None]
    # if arch is None: assert model_cfg['load_pretrained_from'] in ['dir', None]
    print(f"Constructing {arch} encoder...")

    if model_cfg['channel_vit']:
        encoder = ChannelViTEncoder(**model_cfg['kwargs'])
    else:
        encoder = ViTEncoder(**model_cfg['kwargs'])

    # if model_cfg['load_pretrained_from'] != None:
    #     if model_cfg['load_pretrained_from'] == 'torchgeo':
    #         assert model_cfg['pretrained_ckpt'] is not None
    #         weights = tgm.get_weight(model_cfg['pretrained_ckpt'])
    #         state_dict = tgm.get_model(arch, weights=weights).state_dict()

    #     elif model_cfg['load_pretrained_from'] == 'timm':
    #         state_dict = timm.create_model(arch, pretrained=(model_cfg['load_pretrained_from']=='timm')).state_dict()  

    #     elif model_cfg['load_pretrained_from'] == 'dir':
    #         state_dict = torch.load(model_cfg['pretrained_ckpt'])

    #     try:
    #         encoder.load_state_dict(state_dict, strict=False)
    #     except:
    #         raise ValueError(f"Pretrained ckpt for {arch} mismatch with model config")

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

def construct_downstream_models(cfg, modals=['OPTICAL', 'RADAR']):
    model_cfg = cfg.MODEL
    task_head_kwargs = cfg.DATASET['task_head_kwargs']
    arch = model_cfg['architecture']
    models = {}
    states_dicts = get_pretrained_weight(cfg, arch)
    for modal in modals:
        if not hasattr(model_cfg, modal): continue
        modal_cfg = model_cfg[modal]
        assert modal_cfg['use_head']
        encoder = construct_encoder(modal_cfg, arch=arch)
        if states_dicts is not None:
            print(f"Loading pretrained weights for {modal} encoder...")
            encoder_state_dict = states_dicts[modal]
            encoder.load_state_dict(encoder_state_dict, strict=False)

        num_features = encoder.num_features
        head_kwargs = task_head_kwargs
        head_kwargs['use_bias'] = modal_cfg['head_kwargs']['use_bias']
        head_kwargs['in_features'] = num_features

        head = construct_head(head_kwargs)
        model = ViTModel(encoder, head)
        models[modal] = model
    return models

def construct_head(head_cfg):
    if head_cfg['head_type'] == 'linear':
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