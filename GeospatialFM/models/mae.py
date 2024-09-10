import torch.nn as nn
import numpy as np
import torch
from transformers import PreTrainedModel
from .multi_modal_low_rank_vit import MultiModalLowRankViTEncoder, MultiModalLowRankViTDecoder, MultiModalLowRankViTConfig


SENTINEL2_WV = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4]
SENTINEL1_WV = [1575.4, 1660.0] # TODO: fix this

class MultiModalMAEViT(PreTrainedModel):
    config_class = MultiModalLowRankViTConfig
    def __init__(self, config):
        super().__init__(config)
        self.encoder = MultiModalLowRankViTEncoder(config)
        self.decoder = MultiModalLowRankViTDecoder(config)

    def _forward(self, optical, radar, optical_channel_wv=SENTINEL2_WV, radar_channel_wv=SENTINEL1_WV, mask_ratio=0.5, channel_mask_ratio=0.5, spatial_resolution=10, prefix=''):
        latent, channel_mask, channel_ids_restore, pos_mask, pos_ids_restore = self.encoder(optical=optical, radar=radar, optical_channel_wv=optical_channel_wv, radar_channel_wv=radar_channel_wv, 
                                                                       spatial_resolution=spatial_resolution, mask_ratio=mask_ratio, channel_mask_ratio=channel_mask_ratio)
        recon = self.decoder(latent, pos_ids_restore, channel_ids_restore, optical_channel_wv, radar_channel_wv, spatial_resolution, restore_input_dim=False)

        # return dict
        return_dict= {f'{prefix}_channel_mask': channel_mask, 
                      f'{prefix}_recon': recon, 
                      f'{prefix}_pos_mask': pos_mask
                      }
        return return_dict
    
    def forward(self, optical, radar, optical_channel_wv=SENTINEL2_WV, radar_channel_wv=SENTINEL1_WV, mask_ratio=0.5, channel_mask_ratio=0.5, spatial_resolution=10, modal=None):
        assert modal in ['multi', 'optical', 'radar', None]
        optical_target = self.decoder.forward_target(optical)
        radar_target = self.decoder.forward_target(radar)
        target = torch.cat([optical_target, radar_target], dim=1) # B C HW patch_size**2
        return_dict = dict(target=target)
        n_optical_channels = optical_channel_wv.shape[1]
        n_radar_channels = radar_channel_wv.shape[1]
        if modal is None or modal == 'optical':
            optical_dict = self._forward(optical, None, optical_channel_wv, radar_channel_wv, mask_ratio, channel_mask_ratio, spatial_resolution, prefix='optical')
            optical_channel_mask = optical_dict['optical_channel_mask']
            radar_channel_mask = torch.ones(optical_channel_mask.shape[0], n_radar_channels).to(optical_channel_mask.device)
            optical_dict['optical_channel_mask'] = torch.cat([optical_channel_mask, radar_channel_mask], dim=1)
            return_dict.update(optical_dict)
        if modal is None or modal == 'radar':
            radar_dict = self._forward(None, radar, optical_channel_wv, radar_channel_wv, mask_ratio, channel_mask_ratio, spatial_resolution, prefix='radar')
            radar_channel_mask = radar_dict['radar_channel_mask']
            optical_channel_mask = torch.ones(radar_channel_mask.shape[0], n_optical_channels).to(radar_channel_mask.device)
            radar_dict['radar_channel_mask'] = torch.cat([optical_channel_mask, radar_channel_mask], dim=1)
            return_dict.update(radar_dict)
        if modal is None or modal == 'multi':
            multi_dict = self._forward(optical, radar, optical_channel_wv, radar_channel_wv, mask_ratio, channel_mask_ratio, spatial_resolution, prefix='multi')
            return_dict.update(multi_dict)
        return return_dict
