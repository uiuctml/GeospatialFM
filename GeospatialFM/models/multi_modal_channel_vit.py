import torch.nn as nn
from torch.nn.init import trunc_normal_
import random
from torch.nn import functional as F
import torch
from .vision_transformer import *
from typing import Optional
from .patch_embed import PatchEmbedPerChannel
from copy import deepcopy

SENTINEL_WV = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4]

class MultiModalChannelViTEncoder(ViTEncoder):
    """ Multi-Modal_Channel_VisionTransformer backbone that supports Masked Autoencoder
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 optical_in_chans=3,
                 radar_in_chans=2,
                 embed_dim=768,
                 channel_pool='max',
                 sptial_spectral_blocks=0,
                 spectral_blocks=0,
                 **kwargs):
        super().__init__(img_size, patch_size, optical_in_chans, embed_dim, **kwargs) # TODO: check this
        # Additional args for Channel_ViT
        self.channel_pool = channel_pool
        # override the patch embedding
        del self.patch_embed
        self.optical_patch_embed = PatchEmbedPerChannel(img_size, patch_size, optical_in_chans, embed_dim, continuous_channels=True)
        self.radar_patch_embed = PatchEmbedPerChannel(img_size, patch_size, radar_in_chans, embed_dim, continuous_channels=False, enable_sample=False)
        if channel_pool == "mean":
            self.channel_pool = nn.AdaptiveAvgPool1d(1)
        elif channel_pool == "max":
            self.channel_pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise NotImplementedError
        assert spectral_blocks + sptial_spectral_blocks <= self.n_blocks, "number of total blocks should be less than the number of layers"
        self.spectral_blocks = spectral_blocks
        self.sptial_spectral_blocks = sptial_spectral_blocks
        self.spatial_blocks = self.n_blocks - spectral_blocks - sptial_spectral_blocks
        print(f"Spectral Blocks: {self.spectral_blocks}\tSptial-Spectral Blocks: {self.sptial_spectral_blocks}\tSpatial Blocks: {self.spatial_blocks}")
        self.radar_spectral_blocks = deepcopy(self.blocks[:spectral_blocks]) if self.spectral_blocks > 0 else None
        self.optical_spectral_blocks = self.blocks[:spectral_blocks] if self.spectral_blocks > 0 else None

    def _spectral2spatial(self, x, B):
        """Convert from spectral order to spatial order"""
        BHW, Cin, Cout = x.shape
        x = x.view(B, -1, Cin, Cout).permute(0, 2, 1, 3) # B Cin HW Cout
        return x

    def _spatial2spectral(self, x):
        """Convert from spatial order to spectral order"""
        B, Cin, L, Cout = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, Cin, Cout) # BL Cin Cout
        return x
    
    def _pool_channel(self, x):
        B, Cin, HW, Cout = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, Cin) # B HWCout Cin
        x = self.channel_pool(x).reshape(B, HW, Cout) # B HW Cout
        return x
    
    def _pool_channel_with_extra_tokens(self, x, cin):
        post_tokens = x[:, :self.num_register_tokens+1]
        patch_tokens = x[:, self.num_register_tokens+1:] # [B, CinL, Cout]
        # patch_tokens = x
        B, _, Cout = patch_tokens.shape
        patch_tokens = patch_tokens.view(B, cin, -1, Cout) # [B, Cin, L, Cout]
        patch_tokens = self._pool_channel(patch_tokens) # [B, L, Cout]
        x = torch.cat((post_tokens, patch_tokens), dim=1) # [B, L+T, Cout]
        # x = patch_tokens
        return x

    # ViT Forward path
    def forward(self, optical=None, radar=None, channel_ids=SENTINEL_WV, return_dict=False):
        assert optical is not None or radar is not None, "At least one of optical and radar should be provided"
        # embed patches
        if radar is not None:
            radar, _, _ = self.radar_patch_embed(radar, 0, None) # B Cin HW Cout
            channel_mask = None
            radar_Cin = radar.shape[1]
        if optical is not None:
            optical, _channel_mask, channel_ids_restore = self.optical_patch_embed(optical, 0, channel_ids) # B Cin HW Cout
            channel_mask = _channel_mask.unsqueeze(1).expand(-1, self.patch_size**2, -1).flatten(1) 
            optical_Cin = optical.shape[1]

        # temperary concat
        x = self.maybe_concat(optical, radar)

        B, Cin, HW, Cout = x.shape
        if self.spatial_blocks == self.n_blocks:
            assert self.sptial_spectral_blocks == 0, "No spectral blocks are allowed"
            assert self.spectral_blocks == 0, "No spectral blocks are allowed"
            # Normal ViT
            x = self._pool_channel(x)
            pos_embed = self.pos_embed[:, 1:, :]
        else:
            pos_embed = self.interpolate_positional_encoder(Cin)
        x = x + pos_embed # B Cin HW Cout
        L = x.shape[-2]

        # spectral only blocks
        if self.spectral_blocks > 0:
            assert x.dim() == 4, "Input tensor should be B Cin L Cout"
            x = self._spatial2spectral(x) # B Cin HW Cout -> # BHW Cin Cout

            # undo concatenation
            if optical is not None and radar is not None:
                optical = x[:, :optical_Cin]
                radar = x[:, optical_Cin:]
            elif optical is not None:
                optical = x
            elif radar is not None:
                radar = x

            if optical is not None:
                for blk in self.blocks[:self.spectral_blocks]:
                    optical = blk(optical) # BHW Cin Cout
            if radar is not None:
                for blk in self.radar_spectral_blocks:
                    radar = blk(radar) # BHW Cin Cout
            # for blk in self.blocks[:self.spectral_blocks]:
            #     x = blk(x)

            # redo concatenation
            # from now on radar and obtical should be mixed
            x = self.maybe_concat(optical, radar) # BHW Cin Cout
            x = self._spectral2spatial(x, B) # BHW Cin Cout -> B Cin HW Cout

        # from now on x should only have 3 dims    
        if x.dim() == 4:
            x = x.reshape(B, -1, Cout) # B Cin L Cout -> B CinL Cout

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            ) # B CinHW+T Cout

        if self.sptial_spectral_blocks > 0:
            assert x.dim() == 3, "Input tensor should be B L Cout"
            assert x.shape[1] == L * Cin + 1 + self.num_register_tokens
            for blk in self.blocks[self.spectral_blocks:self.spectral_blocks+self.sptial_spectral_blocks]:
                x = blk(x)

        if self.spatial_blocks != self.n_blocks:
            x = self._pool_channel_with_extra_tokens(x, Cin) # B HW+T Cout

        # # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1) 
        
        # if self.register_tokens is not None:
        #     x = torch.cat(
        #         (
        #             x[:, :1],
        #             self.register_tokens.expand(x.shape[0], -1, -1),
        #             x[:, 1:],
        #         ),
        #         dim=1,
        #     ) # B CinHW+T Cout

        if self.spatial_blocks > 0:
            assert x.dim() == 3, "Input tensor should be B L Cout"
            assert x.shape[1] == L + 1 + self.num_register_tokens
            for blk in self.blocks[self.spectral_blocks+self.sptial_spectral_blocks:]:
                x = blk(x)

        x = self.norm(x)

        cls_token = x[:, 0]
        register_tokens = x[:, 1:self.num_register_tokens+1] if self.num_register_tokens else None
        patch_tokens = x[:, self.num_register_tokens+1:]

        if not self.channel_pool:
            raise NotImplementedError

        if return_dict:
            return dict(cls_token=cls_token, register_tokens=register_tokens, patch_tokens=patch_tokens, latent=x)
        return x, cls_token, register_tokens, patch_tokens

    def interpolate_positional_encoder(self, cin):
        pos_embed = self.pos_embed[:, 1:, :] # 1 HW Cout
        pos_embed = pos_embed.unsqueeze(1).repeat(1, cin, 1, 1).permute(0, 3, 1, 2) # 1 Cout Cin HW
        # pos_embed = pos_embed.flatten(2).transpose(1, 2) # 1 CinHW Cout
        pos_embed = pos_embed.permute(0, 2, 3, 1) # 1 Cin HW Cout
        return pos_embed

    def maybe_concat(self, x, y):
        if x is not None and y is not None:
            return torch.cat((x, y), dim=1)
        elif x is not None:
            return x
        elif y is not None:
            return y

    # Encoder for MAE
    def forward_encoder(self, optical=None, radar=None, mask_ratio=0.75, channel_mask_ratio=0.5, channel_ids=SENTINEL_WV, return_dict=False):
        assert optical is not None or radar is not None, "At least one of optical and radar should be provided"
        # embed patches
        if radar is not None:
            radar, _, _ = self.radar_patch_embed(radar, 0, None) # B Cin HW Cout
            channel_mask = None
            radar_Cin = radar.shape[1]
        if optical is not None:
            optical, _channel_mask, channel_ids_restore = self.optical_patch_embed(optical, channel_mask_ratio, channel_ids) # B Cin HW Cout
            channel_mask = _channel_mask.unsqueeze(1).expand(-1, self.patch_size**2, -1).flatten(1) 
            optical_Cin = optical.shape[1]

        # temperary concat
        x = self.maybe_concat(optical, radar)

        B, Cin, HW, Cout = x.shape
        if self.spatial_blocks == self.n_blocks:
            assert self.sptial_spectral_blocks == 0, "No spectral blocks are allowed"
            assert self.spectral_blocks == 0, "No spectral blocks are allowed"
            # Normal ViT
            x = self._pool_channel(x)
            pos_embed = self.pos_embed[:, 1:, :]
        else:
            pos_embed = self.interpolate_positional_encoder(Cin)
        x = x + pos_embed # B Cin HW Cout
    
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self._random_masking(x, mask_ratio) # B Cin L Cout / B L Cout
        L = x.shape[-2]

        # spectral only blocks
        if self.spectral_blocks > 0:
            assert x.dim() == 4, "Input tensor should be B Cin L Cout"
            x = self._spatial2spectral(x) # B Cin L Cout -> # BL Cin Cout

            # undo concatenation
            if optical is not None and radar is not None:
                optical = x[:, :optical_Cin]
                radar = x[:, optical_Cin:]
            elif optical is not None:
                optical = x
            elif radar is not None:
                radar = x

            if optical is not None:
                for blk in self.blocks[:self.spectral_blocks]:
                    optical = blk(optical) # BHW Cin Cout
            if radar is not None:
                for blk in self.radar_spectral_blocks:
                    radar = blk(radar) # BHW Cin Cout
            # for blk in self.blocks[:self.spectral_blocks]:
            #     x = blk(x)

            # redo concatenation
            # from now on radar and obtical should be mixed
            x = self.maybe_concat(optical, radar) # BHW Cin Cout
            x = self._spectral2spatial(x, B) # BHW Cin Cout -> B Cin L Cout

        # from now on x should only have 3 dims    
        if x.dim() == 4:
            x = x.reshape(B, -1, Cout) # B Cin L Cout -> B CinL Cout

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # append register tokens
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        if self.sptial_spectral_blocks > 0:
            assert x.dim() == 3, "Input tensor should be B L Cout"
            assert x.shape[1] == L * Cin + 1 + self.num_register_tokens
            for blk in self.blocks[self.spectral_blocks:self.spectral_blocks+self.sptial_spectral_blocks]:
                x = blk(x)
        
        if self.spatial_blocks != self.n_blocks:
            x = self._pool_channel_with_extra_tokens(x, Cin) # B L+T Cout

        if self.spatial_blocks > 0:
            assert x.dim() == 3, "Input tensor should be B L Cout"
            assert x.shape[1] == L + 1 + self.num_register_tokens
            for blk in self.blocks[self.spectral_blocks+self.sptial_spectral_blocks:]:
                x = blk(x)

        x = self.norm(x)
            
        if return_dict:
            return dict(latent=x, mask=mask, ids_restore=ids_restore, 
                        channel_mask=channel_mask,) 
                        # channel_ids_restore=channel_ids_restore, kept_channels=cin)
        return x, mask, ids_restore, channel_mask#, channel_ids_restore, cin

    def _random_masking(self, x, mask_ratio):
        if x.dim() == 3:
            return self.random_masking(x, mask_ratio)
        elif x.dim() == 4:
            return self.random_masking_3D(x, mask_ratio)
        else:
            raise ValueError("Input tensor should be 3D or 4D")

    def random_masking_3D(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, C, L, D], sequence
        """
        N, C, L, D = x.shape  # batch, channels, length, dim
        x = x.permute(0, 2, 3, 1).flatten(2)  # [N, L, D*C]
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D*C))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = x_masked.view(N, len_keep, D, C).permute(0, 3, 1, 2)  # [N, C, L, D]

        return x_masked, mask, ids_restore