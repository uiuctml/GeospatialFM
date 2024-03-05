import os
import torch
import torchgeo.models as tgm
import torch.nn as nn
import timm
from .vision_transformer import ViTEncoder, ViTDecoder
from .flexible_channel_vit import ChannelViTEncoder
from .multi_modal_channel_vit import MultiModalChannelViTEncoder, MultiModalChannelViTDecoder
from .pspnet import *
from .mae import CrossModalMAEViT
from .multi_modal_mae import MultiModalMAEViT
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class ViTModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        x = self.encoder(x, return_dict=True)['cls_token']
        x = self.head(x)
        return x
    
class ViTMMModel(ViTModel):
    def __init__(self, encoder, head):
        super().__init__(encoder, head)
    
    def forward(self, x):
        assert len(x) == 2
        x = self.encoder(*x, return_dict=True)['cls_token']
        x = self.head(x)
        return x

# CHANGE
class ViTCDModel(ViTModel):
    def __init__(self, encoder, head):
        super().__init__(encoder, head)
    
    def forward(self, x1, x2=None):
        if isinstance(x1, tuple):
            x1, x2 = x1
        else:
            assert x2 is not None, 'image2 must be provided'
        x = self.encoder(x1, x2) 
        x = self.head(x)
        return x
        
def unwrap_model(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
    return new_state_dict

def decompose_model(state_dict, target_modal='optical'):
    # for cross_modal models only
    optical_model = OrderedDict()
    radar_model = OrderedDict()
    models = {'optical': optical_model, 'radar': radar_model}
    for key, value in state_dict.items():
        try:
            key_prefix, key_name = key.split('.', 1)
        except:
            print(key)
            continue
        if 'optical_encoder' in key_prefix:
            models['optical'][key_name] = value
        elif 'radar_encoder' in key_prefix:
            models['radar'][key_name] = value
    return models[target_modal]

def tailor_model(state_dict, handle_modal='cross_modal', target_modal='optical'):
    print(f'The model hanles modality by {handle_modal}...')
    print(f'Targeting {target_modal} encoder...')
    if handle_modal == 'multi_modal':
        target_key = 'encoder'
    else:
        target_key = f'{target_modal}_encoder'
    model = OrderedDict()
    for key, value in state_dict.items():
        try:
            key_prefix, key_name = key.split('.', 1)
        except:
            continue
        if target_key in key_prefix:
            # if handle_modal == 'multi_modal' and target_modal != 'multi_modal':
            #     if f'{target_modal}_spectral_blocks' in key_name:
            #         name = key_name.replace(f'{target_modal}_spectral_blocks.', '')
            #         model[f'blocks.{name}'] = value
            #     elif model.get(key_name) is not None:
            #         continue
            #     else:
            #         model[key_name] = value
            # else:      
            #     model[key_name] = value
            model[key_name] = value
            
    return model

def get_pretrained_weight(cfg, arch, skip_modules=['patch_embed.proj.weight', 'patch_embed.proj.bias'], handle_modal='cross_modal', target_modal='optical'):
    model_cfg = cfg.MODEL
    load_pretrained_from = model_cfg['load_pretrained_from']
    pretrained_ckpt = model_cfg['pretrained_ckpt'] if model_cfg['pretrained_ckpt'] is not None else 'final_model.pth'
    assert load_pretrained_from in ['timm', 'torchgeo', 'dir', None]
    if load_pretrained_from is None:
        return None
    elif load_pretrained_from == 'timm':
        print(f"Loading pretrained weights from {arch}.{pretrained_ckpt}...")
        state_dict = timm.create_model(f'{arch}.{pretrained_ckpt}', pretrained=True).state_dict()
        for skip_module in skip_modules:
            del state_dict[skip_module]
    elif load_pretrained_from == 'dir':
        save_path = os.path.join(cfg.TRAINER['ckpt_dir'], pretrained_ckpt)
        print(f"Loading pretrained weights from {save_path}...")
        state_dict = unwrap_model(torch.load(save_path, map_location='cpu'))
        state_dict = tailor_model(state_dict, handle_modal=handle_modal, target_modal=target_modal)
        # if handle_modal == 'cross_modal':
        #     state_dict = decompose_model(state_dict, target_modal=target_modal)
        # else:
        #     if target_modal == 'multi_modal':
        #         state_dict = decompose_model(state_dict, target_modal='optical')
        # if MM_model:
        #     mm_state_dict = decompose_model(state_dict, MM_model=True)
        #     state_dicts = {'MULTI_MODAL': mm_state_dict}
        # else:
        #     optical_state_dict, radar_state_dict = decompose_model(state_dict)
        #     state_dicts = {'OPTICAL': optical_state_dict, 'RADAR': radar_state_dict}
    else:
        raise NotImplementedError
    return state_dict

def construct_encoder(model_cfg, arch):
    # assert model_cfg['load_pretrained_from'] in ['timm', 'torchgeo', 'dir', None]
    # if arch is None: assert model_cfg['load_pretrained_from'] in ['dir', None]
    print(f"Constructing {arch} encoder...")

    if model_cfg['custom_vit'] == 'channel_vit':
        encoder = ChannelViTEncoder(**model_cfg['kwargs'])
    elif model_cfg['custom_vit'] == 'multi_modal_channel_vit':
        encoder = MultiModalChannelViTEncoder(**model_cfg['kwargs'])
    else:
        encoder = ViTEncoder(**model_cfg['kwargs'])

    if model_cfg['freeze_encoder']:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def construct_decoder(decoder_cfg, channel_decoder=False):
    if channel_decoder:
        return MultiModalChannelViTDecoder(**decoder_cfg['kwargs'])
    else:
        return ViTDecoder(**decoder_cfg['kwargs'])

def construct_mae(model_cfg):
    arch = model_cfg['architecture']

    if model_cfg['handle_modal'] == 'cross_modal':
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

    elif model_cfg['handle_modal'] == 'multi_modal':
        encoder = construct_encoder(model_cfg['MULTI_MODAL'], arch=arch)
        decoder = construct_decoder(model_cfg['DECODER'], channel_decoder=True)
        mae = MultiModalMAEViT(encoder, decoder)
        if model_cfg['MULTI_MODAL']['use_head']:
            head = construct_head(model_cfg['MULTI_MODAL']['head_kwargs'])
            mae.set_head(head)

    else:
        raise NotImplementedError
    return mae

def construct_downstream_models(cfg, target_modal='optical'):
    assert target_modal in ['optical', 'radar', 'multi_modal'], f'target_modal cannot be {target_modal}...'
    model_cfg = cfg.MODEL
    task_head_kwargs = cfg.DATASET['task_head_kwargs']
    arch = model_cfg['architecture']
    state_dict = get_pretrained_weight(cfg, arch, handle_modal=model_cfg['handle_modal'], target_modal=target_modal)

    if model_cfg['handle_modal'] == 'multi_modal':
        config_key = 'MULTI_MODAL'
    else:
        config_key = target_modal.upper()
    assert hasattr(model_cfg, config_key)
    modal_cfg = model_cfg[config_key]
    assert modal_cfg['use_head']
    encoder = construct_encoder(modal_cfg, arch=arch)
    print(f"Loading pretrained weights for {target_modal} encoder...")
    encoder.load_state_dict(state_dict, strict=False)

    if model_cfg['handle_modal'] == 'multi_modal':
        if target_modal=='optical':
            del encoder.radar_spectral_blocks
            del encoder.radar_patch_embed
        elif target_modal=='radar':
            del encoder.optical_spectral_blocks
            del encoder.optical_patch_embed   

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f'Number of parameters for Encoder: {num_params}')

    num_features = encoder.num_features
    head_kwargs = task_head_kwargs
    head_kwargs['use_bias'] = modal_cfg['head_kwargs']['use_bias']
    head_kwargs['in_features'] = num_features

    head = construct_head(head_kwargs)
    if target_modal == 'multi_modal':
        model = ViTMMModel(encoder, head)
    if cfg.DATASET['task_type'] == 'change_detection':
        encoder = CDEncoder(encoder, diff=True, use_mlp=head_kwargs['use_mlp'])
        encoder.requires_grad = False
        model = ViTCDModel(encoder, head)
    else:  
        model = ViTModel(encoder, head)
    return model

def construct_head(head_cfg):
    print(f"Constructing {head_cfg['head_type']} head...")
    if head_cfg['head_type'] == 'linear':
        head = nn.Linear(head_cfg['in_features'], head_cfg['num_classes'], bias=head_cfg['use_bias'])
    elif head_cfg['head_type'] == 'pspnet':
        head = PSPNetDecoder(head_cfg['in_features'], head_cfg['num_classes'], head_cfg['hidden_dim'], head_cfg['image_size'])
    else:
        raise NotImplementedError
    return head