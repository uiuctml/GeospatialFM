# from .vision_transformer import ViTEncoder, ViTDecoder
import torch.nn as nn
import numpy as np
import torch

class CrossModalMAEViT(nn.Module):
    def __init__(self, 
                 optical_encoder, 
                 radar_encoder,
                 optical_decoder,
                 radar_decoder,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None,
                 ):
        
        super().__init__()
        self.optical_encoder = optical_encoder
        self.radar_encoder = radar_encoder
        self.optical_decoder = optical_decoder
        self.radar_decoder = radar_decoder

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale).requires_grad_(False)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias).requires_grad_(False)
        else:
            self.logit_bias = None

    def set_head(self, optical_head, radar_head):
        self.optical_head = optical_head
        self.radar_head = radar_head

    def forward_recon(self, optical, radar, mask_ratio=0.75):
        # forward optical
        optical_latent, optical_mask, optical_ids_restore = self.optical_encoder.forward_encoder(optical, mask_ratio)
        optical_recon = self.optical_decoder.forward_decoder(optical_latent, optical_ids_restore, restore_input_dim=True)
        optical_cls_token = optical_latent[:, 0]
        # forward radar
        radar_latent, radar_mask, radar_ids_restore = self.radar_encoder.forward_encoder(radar, mask_ratio)
        radar_recon = self.radar_decoder.forward_decoder(radar_latent, radar_ids_restore, restore_input_dim=True)
        radar_cls_token = radar_latent[:, 0]
        return_dict= dict(optical_cls_token=optical_cls_token, radar_cls_token=radar_cls_token,
                    optical_recon=optical_recon, radar_recon=radar_recon,
                    optical_mask=optical_mask, radar_mask=radar_mask,)
        return return_dict

    def forward(self, optical, radar, mask_ratio=0.75):
        # forward optical
        optical_latent, optical_mask, optical_ids_restore = self.optical_encoder.forward_encoder(optical, mask_ratio)
        optical_recon = self.optical_decoder.forward_decoder(optical_latent, optical_ids_restore)
        optical_target = self.optical_decoder.forward_target(optical)
        optical_cls_token = optical_latent[:, 0]
        # forward radar
        radar_latent, radar_mask, radar_ids_restore = self.radar_encoder.forward_encoder(radar, mask_ratio)
        radar_recon = self.radar_decoder.forward_decoder(radar_latent, radar_ids_restore)
        radar_target = self.radar_decoder.forward_target(radar)
        radar_cls_token = radar_latent[:, 0]
        # return dict
        return_dict= dict(optical_mask=optical_mask, radar_mask=radar_mask,
                    optical_recon=optical_recon, radar_recon=radar_recon,
                    optical_target=optical_target, radar_target=radar_target,
                    optical_cls_token=optical_cls_token, radar_cls_token=radar_cls_token,
                    logit_scale=self.logit_scale.exp())
        # downstream head
        if hasattr(self, 'optical_head'):
            optical_logits = self.optical_head(optical_cls_token)
            return_dict['optical_logits'] = optical_logits
        if hasattr(self, 'radar_head'):
            radar_logits = self.radar_head(radar_cls_token)
            return_dict['radar_logits'] = radar_logits
        return return_dict
