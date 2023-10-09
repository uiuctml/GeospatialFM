from .utils import *
from .modeling import *
import os.path as osp
import timm



# def construct_model(model_cfg):
#     assert model_cfg['load_pretrained_from'] in ['timm', 'torchgeo', 'dir', None]
#     criterion = get_criterion(model_cfg['criterion'])

#     if model_cfg['load_pretrained_from'] == 'dir':
#         return EncoderDecoder.load(model_cfg['pretrained_ckpt'])
    
#     elif model_cfg['load_pretrained_from'] == 'torchgeo':
#         assert model_cfg['pretrained_ckpt'] is not None
#         weights = tgm.get_weight(model_cfg['pretrained_ckpt'])
#         encoder = tgm.get_model(model_cfg['architecture'], weights=weights)
#         if model_cfg['architecture'].startswith('vit'):
#             embed_dim = encoder.head.in_features
#             encoder.head = nn.Identity()
#         elif model_cfg['architecture'].startswith('resnet'):
#             embed_dim = encoder.fc.in_features
#             encoder.fc = nn.Identity()

#     else:
#         encoder = timm.create_model(model_cfg['architecture'], pretrained=(model_cfg['load_pretrained_from']=='timm'))
#         if model_cfg['architecture'].startswith('vit'):
#             conv_in = encoder.patch_embed.proj
#         elif model_cfg['architecture'].startswith('resnet'):
#             conv_in = encoder.conv1

#         kernel_size = conv_in.kernel_size[0]
#         stride = conv_in.stride[0]
#         padding = conv_in.padding[0]
#         out_channels = conv_in.out_channels
#         bias = conv_in.bias is not None

#         # brute force duplicate the weights for each band
#         weight = conv_in.weight.data
#         _weight = weight.repeat(1, model_cfg['bands']//weight.shape[1]+1, 1, 1)[:, :model_cfg['bands'], :, :]
#         _bias = conv_in.bias.data if bias else None

#         band_ext_conv_in = nn.Conv2d(model_cfg['bands'], out_channels, kernel_size, stride, padding, bias=bias)
#         band_ext_conv_in.weight.data = _weight
#         band_ext_conv_in.bias.data = _bias
#         fc_out = nn.Identity()

#         if model_cfg['architecture'].startswith('vit'):
#             encoder.patch_embed.proj = band_ext_conv_in
#             embed_dim = encoder.head.in_features
#             encoder.head = fc_out
#         elif model_cfg['architecture'].startswith('resnet'):
#             encoder.conv1 = band_ext_conv_in
#             embed_dim = encoder.fc.in_features
#             encoder.fc = conv_in

#     if model_cfg['head_extra_kwargs']['head_type'] == 'linear':
#         task_head = ClassificationHead(out_features=model_cfg['num_classes'], in_features=embed_dim, **model_cfg['head_extra_kwargs'])
#     model = EncoderDecoder(encoder, task_head, lp=model_cfg['lp'], criterion=criterion)

#     return model