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

def collpase_channels(x, reduce=None):
    assert len(x.shape) == 3
    if reduce == "mean":
        return x.mean(dim=1)
    elif reduce == "sum":
        return x.sum(dim=1)
    elif reduce == "flatten":
        return x.view(x.shape[0], -1, x.shape[-1])
    elif reduce is None:
        return x
    else:
        raise NotImplementedError

class ChannelAttention(nn.Module):
    def __init__(self, embed_dim):
        super(ChannelAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # Assuming x is of shape [B, C, E]
        # Compute query, key, value matrices
        query = self.query(x)  # Shape [B, C, E]
        key = self.key(x)      # Shape [B, C, E]
        value = self.value(x)  # Shape [B, C, E]
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # Shape [B, C, C]
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # Apply attention scores to value matrix
        weighted_values = torch.matmul(attention_scores, value)  # Shape [B, C, E]
        
        # Combine weighted values for all channels
        combined = weighted_values.sum(dim=1)  # Shape [B, E]
        
        return combined


class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        channel_pool: Optional[str] = None,
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
            cin = len_keep
        else:
            mask = torch.zeros([1, Cin], device=x.device)
            cin = Cin
            ids_restore = torch.arange(Cin, device=x.device) # TODO: check this

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        x += cur_channel_embed.unsqueeze(-1).unsqueeze(-1)
        # x += self.channel_embed[:, :, cur_channels, :, :]  # B Cout Cin H W

        # expand mask and ids_restore to B
        mask = mask.expand(B, -1)
        ids_restore = ids_restore.expand(B, -1)
        return x, mask, ids_restore, cin


class ChannelViTEncoder(ViTEncoder):
    """ Channel_VisionTransformer backbone that supports Masked Autoencoder
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 channel_pool='max',
                 pool_position=0,
                 **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, **kwargs) # TODO: check this
        # Additional args for Channel_ViT
        self.channel_pool = channel_pool
        # override the patch embedding
        self.patch_embed = PatchEmbedPerChannel(img_size, patch_size, in_chans, embed_dim, channel_pool=self.channel_pool)
        if channel_pool == "mean":
            self.channel_pool = nn.AdaptiveAvgPool1d(1)
        elif channel_pool == "max":
            self.channel_pool = nn.AdaptiveMaxPool1d(1)
        elif channel_pool == "attention":
            print("Using Channel Attention")
            self.channel_pool = ChannelAttention(embed_dim)
        else:
            raise NotImplementedError
        assert pool_position in [0, 1]
        self.pool_position = pool_position
        
    # ViT Forward path
    def forward(self, x, channel_ids=SENTINEL_WV, return_dict=False):
        x, _, _, _ = self.patch_embed(x, 0, channel_ids)
        if self.pool_position == 0:
            if isinstance(self.channel_pool, ChannelAttention):
                B, Cout, Cin, H, W = x.shape
                x = x.permute(0, 3, 4, 2, 1).reshape(-1, Cin, Cout)
                x = self.channel_pool(x).reshape(B, -1, Cout)
            else:
                # preparing the output sequence
                x = x.flatten(3)  # B Cout Cin HW
                B, Cout, Cin, HW = x.shape
                x = x.permute(0, 3, 1, 2).reshape(B, -1, Cin) # B HWCout Cin
                x = self.channel_pool(x).reshape(B, HW, Cout) # B HW Cout
            
            pos_embed = self.pos_embed[:, 1:, :]
            x = x + pos_embed # B HW Cout 
        else:
            x = x.flatten(3) # B Cout Cin HW
            x = x.permute(0, 2, 3, 1) # B Cin HW Cout

            pos_embed = self.interpolate_positional_encoder(cin) # TODO: check this
            x = x + pos_embed # B Cin HW Cout

        # # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

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
            )

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.pool_position == 1:
            post_tokens = x[:, :self.num_register_tokens+1]
            patch_tokens = x[:, self.num_register_tokens+1:] # [B, CinL, Cout]
            B, _, Cout = patch_tokens.shape
            patch_tokens = patch_tokens.view(B, cin, -1, Cout) # [B, Cin, L, Cout]
            if isinstance(self.channel_pool, ChannelAttention):
                patch_tokens = patch_tokens.permute(0, 2, 1, 3).reshape(-1, cin, Cout)
                patch_tokens = self.channel_pool(patch_tokens).reshape(B, -1, Cout)
            else:
                # preparing the output sequence
                patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(B, -1, Cin) # B LCout Cin
                patch_tokens = self.channel_pool(patch_tokens).reshape(B, -1, Cout) # B HW Cout
            x = torch.cat((post_tokens, patch_tokens), dim=1)

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
        x, _channel_mask, channel_ids_restore, cin = self.patch_embed(x, channel_mask_ratio, channel_ids)
        channel_mask = _channel_mask.unsqueeze(1).expand(-1, self.patch_size**2, -1).flatten(1)
        if self.pool_position == 0:
            if isinstance(self.channel_pool, ChannelAttention):
                B, Cout, Cin, H, W = x.shape
                x = x.permute(0, 3, 4, 2, 1).reshape(-1, Cin, Cout)
                x = self.channel_pool(x).reshape(B, -1, Cout)
            else:
                # preparing the output sequence
                x = x.flatten(3)  # B Cout Cin HW
                B, Cout, Cin, HW = x.shape
                x = x.permute(0, 3, 1, 2).reshape(B, -1, Cin) # B HWCout Cin
                x = self.channel_pool(x).reshape(B, HW, Cout) # B HW Cout

            pos_embed = self.pos_embed[:, 1:, :]
            x = x + pos_embed # B HW Cout 
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)           
        else:
            x = x.flatten(3) # B Cout Cin HW
            x = x.permute(0, 2, 3, 1) # B Cin HW Cout

            pos_embed = self.interpolate_positional_encoder(cin) # TODO: check this
            x = x + pos_embed # B Cin HW Cout
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking_3D(x, mask_ratio)

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

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) 

        if self.pool_position == 1:
            post_tokens = x[:, :self.num_register_tokens+1]
            patch_tokens = x[:, self.num_register_tokens+1:] # [B, CinL, Cout]
            B, _, Cout = patch_tokens.shape
            patch_tokens = patch_tokens.view(B, cin, -1, Cout) # [B, Cin, L, Cout]
            if isinstance(self.channel_pool, ChannelAttention):
                patch_tokens = patch_tokens.permute(0, 2, 1, 3).reshape(-1, cin, Cout)
                patch_tokens = self.channel_pool(patch_tokens).reshape(B, -1, Cout)
            else:
                # preparing the output sequence
                patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(B, -1, cin) # B LCout Cin
                patch_tokens = self.channel_pool(patch_tokens).reshape(B, -1, Cout) # B HW Cout
            x = torch.cat((post_tokens, patch_tokens), dim=1)
            
        if return_dict:
            return dict(latent=x, mask=mask, ids_restore=ids_restore, 
                        channel_mask=channel_mask,) 
                        # channel_ids_restore=channel_ids_restore, kept_channels=cin)
        return x, mask, ids_restore, channel_mask#, channel_ids_restore, cin

    def random_masking_3D(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, C, L, D], sequence
        """
        N, C, L, D = x.shape  # batch, length, dim
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
        x_masked = x_masked.view(N, len_keep, D, C).permute(0, 3, 1, 2).reshape(N, -1, D)  # [N, CL, D]

        return x_masked, mask, ids_restore