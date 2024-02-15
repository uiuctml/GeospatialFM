import torch.nn as nn
from torch.nn.init import trunc_normal_
import random
from torch.nn import functional as F
import torch
from .vision_transformer import *
from .pos_embed import ChannelEmbedding
from typing import Optional

SENTINEL_WV = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4]
# SENTINEL_WV = [4.427, 4.924, 5.598, 6.646, 7.041, 7.405, 7.828, 8.328, 8.647, 9.451, 13.735, 16.137, 22.024]

class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
            
        self.proj = nn.Sequential(
            nn.Conv3d(
                1,
                embed_dim*2, #2
                kernel_size=(1, patch_size, patch_size),
                stride=(1, patch_size, patch_size),
            ),
            nn.Conv3d(
                embed_dim*2,
                embed_dim,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                bias=False,
            ),
        )

        # self.channel_embed = nn.Embedding(in_chans, embed_dim)
        self.channel_embed = ChannelEmbedding(in_chans, embed_dim)  # dynamic but fixed sin-cos embedding

        self.enable_sample = enable_sample

        # trunc_normal_(self.channel_embed.weight, std=0.02)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        try:
            w = self.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        except:
            w1 = self.proj[0].weight.data
            w2 = self.proj[1].weight.data
            torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))

    def forward(self, x, channel_mask_ratio=0.5, channel_ids=SENTINEL_WV):
        # embedding lookup
        if channel_ids is None:
            channel_ids = np.arange(x.shape[1])
        cur_channel_embed = self.channel_embed(channel_ids)  # 1, Cin, embed_dim=Cout
        cur_channel_embed = cur_channel_embed.permute(0, 2, 1)  # 1 Cout Cin

        B, Cin, H, W = x.shape
        if self.training and self.enable_sample:
            len_keep = int(Cin * (1 - channel_mask_ratio))
            noise = torch.rand(1, Cin, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(B, 1, H, W))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([1, Cin], device=x.device)
            # mask = torch.ones([B, Cin], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Update the embedding lookup
            cur_channel_embed = torch.gather(cur_channel_embed, 2, ids_keep.unsqueeze(1).repeat(1, cur_channel_embed.shape[1], 1))
            ######
        else:
            mask = torch.zeros([1, Cin], device=x.device)
            ids_restore = torch.arange(Cin, device=x.device) # TODO: check this

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        x += cur_channel_embed.unsqueeze(-1).unsqueeze(-1)
        x = x.flatten(3).permute(0, 2, 3, 1)  # B Cout Cin HW -> B Cin HW Cout

        # expand mask and ids_restore to B
        mask = mask.expand(B, -1)
        ids_restore = ids_restore.expand(B, -1)
        return x, mask, ids_restore


class ChannelViTEncoder(ViTEncoder):
    """ Channel_VisionTransformer backbone that supports Masked Autoencoder
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 channel_pool='max',
                 sptial_spectral_blocks=0,
                 spectral_blocks=0,
                 **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, **kwargs) # TODO: check this
        # Additional args for Channel_ViT
        self.channel_pool = channel_pool
        # override the patch embedding
        self.patch_embed = PatchEmbedPerChannel(img_size, patch_size, in_chans, embed_dim)
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

    def _pool_channel(self, x):
        B, Cin, HW, Cout = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, -1, Cin) # B HWCout Cin
        x = self.channel_pool(x).reshape(B, HW, Cout) # B HW Cout
        return x

    def _spectral2spatial(self, x, B):
        """Convert from spectral order to spatial order"""
        BHW, Cin, Cout = x.shape
        x = x.view(B, -1, Cin, Cout).permute(0, 2, 1, 3).reshape(B, -1, Cout) # B CinHW Cout
        return x

    def _spatial2spectral(self, x):
        """Convert from spatial order to spectral order"""
        B, Cin, L, Cout = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, Cin, Cout) # BL Cin Cout
        return x

    def _pool_channel_with_extra_tokens(self, x, cin):
        if self.spectral_blocks == 0:
            post_tokens = x.view(x.shape[0], cin, -1, x.shape[2])[:, :, :self.num_register_tokens+1, :].view(x.shape[0], -1, x.shape[2]).permute(0, 2, 1) # if spectral_blocks==0, we have each channel with one cls_token, extract, concat, and then pooling
            post_tokens = self.channel_pool(post_tokens).permute(0, 2, 1)

            patch_tokens = x.view(x.shape[0], cin, -1, x.shape[2])[:, :, self.num_register_tokens+1:, :]
            patch_tokens = patch_tokens.reshape(x.shape[0], -1, x.shape[2])
        else:
            post_tokens = x[:, :self.num_register_tokens+1]
            patch_tokens = x[:, self.num_register_tokens+1:] # [B, CinL, Cout]
        B, _, Cout = patch_tokens.shape
        #patch_tokens = patch_tokens.view(B, cin, -1, Cout).permute(0, 3, 1, 2) # [B, Cin, L, Cout] -> [B, Cout, Cin, L]
        patch_tokens = patch_tokens.view(B, cin, -1, Cout).permute(0, 3, 2, 1) # [B, Cin, L, Cout] -> [B, Cout, L, Cin]
        patch_tokens = patch_tokens.reshape(B, -1, cin) # [B, Cout, L, Cin] -> [B, Cout * L, Cin]
        patch_tokens = self.channel_pool(patch_tokens) # [B, HW, Cout]
        patch_tokens = patch_tokens.view(B, Cout, -1).permute(0, 2, 1) # [B, L, Cout]
        x = torch.cat((post_tokens, patch_tokens), dim=1) # [B, L+T, Cout]
        return x

    # ViT Forward path
    def forward(self, x, channel_ids=SENTINEL_WV, return_dict=False):
        x, _, _ = self.patch_embed(x, channel_mask_ratio=0, channel_ids=channel_ids)
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

        # spectral only blocks
        if self.spectral_blocks > 0:
            assert x.dim() == 4, "Input tensor should be B Cin L Cout"
            x = self._spatial2spectral(x) # B Cin HW Cout -> # BHW Cin Cout
            for blk in self.blocks[:self.spectral_blocks]:
                x = blk(x)
            x = self._spectral2spatial(x, B) # BHW Cin Cout -> B CinHW Cout

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
            x = self._pool_channel_with_extra_tokens(x, Cin) # B HW+T Cout

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

    # Encoder for MAE
    def forward_encoder(self, x, mask_ratio=0.75, channel_mask_ratio=0.5, channel_ids=SENTINEL_WV, return_dict=False):
        # embed patches
        x, _channel_mask, channel_ids_restore = self.patch_embed(x, channel_mask_ratio, channel_ids)
        channel_mask = _channel_mask.unsqueeze(1).expand(-1, self.patch_size**2, -1).flatten(1)

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
    
        x = x + pos_embed # B Cin HW Cout
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self._random_masking(x, mask_ratio) # B Cin L Cout / B L Cout
        L = x.shape[-2]
        num_patch_spectral_spatial = L * Cin

        # spectral only blocks
        if self.spectral_blocks > 0:
            assert x.dim() == 4, "Input tensor should be B Cin L Cout"
            x = self._spatial2spectral(x) # B Cin L Cout -> # BL Cin Cout
            for blk in self.blocks[:self.spectral_blocks]:
                x = blk(x)
            x = self._spectral2spatial(x, B) # BHW Cin Cout -> B CinHW Cout
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2]) # B CinHW Cout -> B 1 CinHW Cout

        # append cls token
        if self.spatial_blocks == self.n_blocks: # spectral_blocks == spatial_spectral_blocks == 0
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        else: 
            if self.sptial_spectral_blocks == 0: # spectral_blocks != 0
                # TODO: could move cls_token cat after channel pooling?
                x = x.view(x.shape[0], Cin, x.shape[2] // Cin, x.shape[3]) # B 1 CinHW Cout -> B Cin HW Cout
                cls_token = self.cls_token + self.pos_embed[:, :1, :]
                cls_tokens = cls_token.expand(x.shape[0], x.shape[1], -1, -1)
                x = torch.cat((cls_tokens, x), dim=2)
                x = self._pool_channel(x) # no spatial_spectral_block afterward, do channel pooling
            else: # spectral == 0 and spatial_spectral_blocks != 0 (i.e. we have the channel dim)
                cls_token = self.cls_token + self.pos_embed[:, :1, :]
                cls_tokens = cls_token.expand(x.shape[0], x.shape[1], -1, -1)
                x = torch.cat((cls_tokens, x), dim=2)

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
            # assert x.dim() == 3, "Input tensor should be B L Cout"
            assert x.dim() == 4, "Input tensor should be B Cin L Cout; if going through specral block, then B 1 L*Cin Cout"
            # assert x.shape[1] == L * Cin + 1 + self.num_register_tokens
            assert x.shape[1] * x.shape[2] == num_patch_spectral_spatial + 1 + self.num_register_tokens or x.shape[1] * x.shape[2] == num_patch_spectral_spatial + 3 + self.num_register_tokens
            x = x.view(x.shape[0], -1, x.shape[3])
            for blk in self.blocks[self.spectral_blocks:self.spectral_blocks+self.sptial_spectral_blocks]:
                x = blk(x)
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