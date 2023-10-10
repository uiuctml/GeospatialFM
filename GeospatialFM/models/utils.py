import os
import torch
import torchgeo.models as tgm
from .modeling import *
import timm

def get_criterion(criterion):
    if criterion == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

def consturct_encoder(model_cfg):
    assert model_cfg['load_pretrained_from'] in ['timm', 'torchgeo', 'dir', None]

    if model_cfg['load_pretrained_from'] == 'dir':
        encoder = torch.load(model_cfg['pretrained_ckpt'])
    
    elif model_cfg['load_pretrained_from'] == 'torchgeo':
        assert model_cfg['pretrained_ckpt'] is not None
        weights = tgm.get_weight(model_cfg['pretrained_ckpt'])
        encoder = tgm.get_model(model_cfg['architecture'], weights=weights)

    else:
        encoder = timm.create_model(model_cfg['architecture'], pretrained=(model_cfg['load_pretrained_from']=='timm'))
    
    if model_cfg['architecture'].startswith('vit'):
        conv_in = encoder.patch_embed.proj
    elif model_cfg['architecture'].startswith('resnet'):
        conv_in = encoder.conv1

    kernel_size = conv_in.kernel_size[0]
    stride = conv_in.stride[0]
    padding = conv_in.padding[0]
    out_channels = conv_in.out_channels
    bias = conv_in.bias is not None

    # brute force duplicate the weights for each band
    weight = conv_in.weight.data
    _weight = weight.repeat(1, model_cfg['bands']//weight.shape[1]+1, 1, 1)[:, :model_cfg['bands'], :, :]
    _bias = conv_in.bias.data if bias else None

    band_ext_conv_in = nn.Conv2d(model_cfg['bands'], out_channels, kernel_size, stride, padding, bias=bias)
    band_ext_conv_in.weight.data = _weight
    if bias:
        band_ext_conv_in.bias.data = _bias

    if model_cfg['architecture'].startswith('vit'):
        encoder.patch_embed.proj = band_ext_conv_in
        encoder.head = nn.Identity()
    elif model_cfg['architecture'].startswith('resnet'):
        encoder.conv1 = band_ext_conv_in
        encoder.fc = nn.Identity()
        
    if model_cfg['freeze_encoder']:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def construct_model(model_cfg):
    criterion = get_criterion(model_cfg['criterion'])
    encoder = consturct_encoder(model_cfg)

    head_cfg = model_cfg['head_kwargs']
    if head_cfg['task_type'] == 'classification':
        head = ClassificationHead(out_features=head_cfg['num_classes'], in_features=head_cfg['in_features'], use_bias=head_cfg['use_bias'])
        model = EncoderDecoder(encoder, head, freeze_encoder=model_cfg['freeze_encoder'], criterion=criterion)
    elif head_cfg['task_type'] == 'segmentation':
        if head_cfg['head_type'] == 'unet':
            assert model_cfg['architecture'].startswith('resnet')
            model = Unet(encoder_name = model_cfg['architecture'],
                         in_channels = model_cfg['bands'],
                         classes = head_cfg['num_classes'],
                         criterion=criterion,
                         freeze_encoder = model_cfg['freeze_encoder'],)
            model.encoder.load_state_dict(encoder.state_dict())
    else:
        raise NotImplementedError

    return model