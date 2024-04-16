import torch.nn as nn
import numpy as np
import torch
from .vision_transformer import ViTEncoder
from .multi_modal_channel_vit import MultiModalChannelViTEncoder

SENTINEL_WV = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4]

class MultiModalMAEViT(nn.Module):
    def __init__(self, 
                 multimodal_encoder, 
                 unimodal_decoder,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 share_patch_embedding=True,
                 ):
        
        super().__init__()
        self.encoder = multimodal_encoder
        self.decoder = unimodal_decoder

        self.logit_scale = torch.ones([]) * init_logit_scale
        self.logit_bias = init_logit_bias

    def set_head(self, head):
        self.head = head

    def forward_recon(self, optical=None, radar=None, mask_ratio=0.75, slice_patch_tokens=None, channel_mask_ratio=0.5, channel_ids=SENTINEL_WV):
        latent, mask, ids_restore, optical_channel_mask = self.encoder.forward_encoder(optical=optical, radar=radar, mask_ratio=mask_ratio, channel_mask_ratio=channel_mask_ratio, channel_ids=channel_ids)
        recon = self.decoder.forward_decoder(latent, ids_restore, restore_input_dim=True, slice_patch_tokens=slice_patch_tokens)
        cls_token = latent[:, 0]

        return_dict= dict(cls_token=cls_token, recon=recon, mask=mask)
        if optical_channel_mask is not None:
            return_dict['optical_channel_mask'] = optical_channel_mask
        return return_dict

    def _forward(self, optical, radar, mask_ratio=0.75, channel_mask_ratio=0.5, prefix='', channel_ids=SENTINEL_WV):
        latent, mask, ids_restore, optical_channel_mask = self.encoder.forward_encoder(optical=optical, radar=radar, mask_ratio=mask_ratio, channel_mask_ratio=channel_mask_ratio, channel_ids=channel_ids)
        recon = self.decoder.forward_decoder(latent, ids_restore)
        cls_token = latent[:, 0]

        # return dict
        return_dict= {f'{prefix}mask': mask, 
                      f'{prefix}recon': recon, 
                      f'{prefix}cls_token': cls_token
                      }
        if optical_channel_mask is not None:
            return_dict[f'{prefix}channel_mask'] = optical_channel_mask
        return return_dict
    
    def forward(self, optical, radar, mask_ratio=0.75, channel_mask_ratio=0.5, modal=None, channel_ids=SENTINEL_WV):
        assert modal in ['multi', 'optical', 'radar', None]
        optical_target = self.decoder.forward_target(optical)
        radar_target = self.decoder.forward_target(radar)
        return_dict = dict(optical_target=optical_target, radar_target=radar_target, logit_scale=self.logit_scale.exp())
        if modal is None or modal == 'optical':
            optical_dict = self._forward(optical, None, mask_ratio, channel_mask_ratio, prefix='optical_', channel_ids=channel_ids)
            return_dict.update(optical_dict)
        if modal is None or modal == 'radar':
            radar_dict = self._forward(None, radar, mask_ratio, channel_mask_ratio, prefix='radar_', channel_ids=channel_ids)
            return_dict.update(radar_dict)
        if modal is None or modal == 'multi':
            multi_dict = self._forward(optical, radar, mask_ratio, channel_mask_ratio, prefix='multi_', channel_ids=channel_ids)
            return_dict.update(multi_dict)
        return return_dict
