import torch.nn as nn
from torch.nn.init import trunc_normal_
import random
from torch.nn import functional as F
import torch
from .vision_transformer import *
from typing import Optional

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
        if channel_pool == "mean":
            self.channel_pool = nn.AdaptiveAvgPool1d(1)
        elif channel_pool == "max":
            self.channel_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.channel_pool = None

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )  # CHANGED

        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        self.enable_sample = enable_sample

        trunc_normal_(self.channel_embed.weight, std=0.02)

    def forward(self, x, channel_mask_ratio=0.5, channel_ids=None):
        # embedding lookup
        if channel_ids is None:
            channel_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        cur_channel_embed = self.channel_embed(
            channel_ids #FIXME: better channel embedding
        )  # B, Cin, embed_dim=Cout
        cur_channel_embed = cur_channel_embed.permute(0, 2, 1)  # B Cout Cin

        B, Cin, H, W = x.shape
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans

        if self.training and self.enable_sample:
            len_keep = int(Cin * (1 - channel_mask_ratio))
            # noise = torch.rand(1, Cin, device=x.device)  # noise in [0, 1]
            noise = torch.rand(B, Cin, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            # x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(B, 1, H, W))
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, Cin], device=x.device)
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

        if self.channel_pool is not None:
            # preparing the output sequence
            x = x.flatten(3)  # B Cout Cin HW
            B, Cout, Cin, HW = x.shape
            x = x.permute(0, 3, 1, 2).reshape(B, -1, Cin) # B HWCout Cin
            x = self.channel_pool(x).reshape(B, HW, Cout) # B HW Cout
            # x = x.reshape(B, Cin, -1)
            # x = x.mean(dim=1)
        else:
            x = x.flatten(2) # B Cout CinHW
            x = x.transpose(1, 2)  # B CinHW Cout
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
                 **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, **kwargs) # TODO: check this
        # Additional args for Channel_ViT
        self.channel_pool = channel_pool
        # override the patch embedding
        self.patch_embed = PatchEmbedPerChannel(img_size, patch_size, in_chans, embed_dim, channel_pool=self.channel_pool)
        
    # ViT Forward path
    def forward(self, x, channel_mask_ratio=0, channel_ids=None, return_dict=False):
        x, _, _, _ = self.patch_embed(x, channel_mask_ratio, channel_ids)
        # if self.collpase_embed:
        #     x = collpase_channels(x, 'mean')

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

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
        pos_embed = pos_embed.flatten(2).transpose(1, 2) # 1 CinHW Cout
        return pos_embed

    # Encoder for MAE
    def forward_encoder(self, x, mask_ratio=0.75, channel_mask_ratio=0.5, channel_ids=None, return_dict=False):
        # embed patches
        x, _channel_mask, channel_ids_restore, cin = self.patch_embed(x, channel_mask_ratio, channel_ids)
        channel_mask = _channel_mask.unsqueeze(1).expand(-1, self.patch_size**2, -1).flatten(1)
        if not self.channel_pool:
            # channel_mask_patch = _channel_mask.unsqueeze(1).expand(-1, self.num_patches_per_channel, -1).flatten(1)
            # keep_idx = torch.argwhere(channel_mask_patch[0] == 0).flatten().unsqueeze(0)
            # pos_embed = torch.gather(self.pos_embed[:, 1:, :], 1, keep_idx.unsqueeze(-1).repeat(1, 1, self.embed_dim))
            pos_embed = self.interpolate_positional_encoder(cin) # TODO: check this
        else:
            pos_embed = self.pos_embed[:, 1:, :]
        # print(x.shape, pos_embed.shape)
        # if self.collpase_embed:
        #     x = collpase_channels(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        x = x + pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

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

        if return_dict:
            return dict(latent=x, mask=mask, ids_restore=ids_restore, 
                        channel_mask=channel_mask,) 
                        # channel_ids_restore=channel_ids_restore, kept_channels=cin)
        return x, mask, ids_restore, channel_mask#, channel_ids_restore, cin


# class ChannelViTDecoder(ViTDecoder):
#     def forward_decoder(self, x, ids_restore, channel_ids_restore, kept_channels, restore_input_dim=False, slice_patch_tokens=None):
#         # embed tokens
#         x = self.decoder_embed(x)

#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         x_ = torch.cat([x[:, 1+self.num_register_tokens:, :], mask_tokens], dim=1)  # no cls token and register tokens
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         # x_ in shape B cinHW Cout # TODO: check this
#         print(x_.shape)
#         x_ = x_.reshape(x_.shape[0], kept_channels, -1, x_.shape[-1]) # B cin HW Cout
#         print(x_.shape)
#         # restore the masked channels
#         # TODO: implement this
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

#         # add pos embed
#         x = x + self.decoder_pos_embed

#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)

#         # predictor projection
#         x = self.decoder_pred(x)

#         # remove cls token
#         x = x[:, 1+self.num_register_tokens:, :]

#         if restore_input_dim:
#             x = self.unpatchify(x, slice_patch_tokens)

#         return x