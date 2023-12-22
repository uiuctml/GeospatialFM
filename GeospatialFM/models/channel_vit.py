import torch.nn as nn
from torch.nn.init import trunc_normal_
import random
from torch.nn import functional as F
import torch
from .vision_transformer import *

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
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

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
        len_keep = int(Cin * (1 - channel_mask_ratio))

        mask = None
        if self.training and self.enable_sample:
            noise = torch.rand(B, Cin, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, Cin], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Update the embedding lookup
            cur_channel_embed = torch.gather(cur_channel_embed, 2, ids_keep.unsqueeze(1).repeat(1, cur_channel_embed.shape[1], 1))
            ######

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        x += cur_channel_embed.unsqueeze(-1).unsqueeze(-1)
        # x += self.channel_embed[:, :, cur_channels, :, :]  # B Cout Cin H W

        # preparing the output sequence
        x = x.flatten(3)  # B Cout Cin HW
        # x = x.transpose(1, 2)  # B CinHW Cout
        x = x.permute(0, 2, 3, 1) # B Cin HW Cout
        x = x.mean(dim=1)

        return x, mask


class ChannelViTEncoder(ViTEncoder):
    """ Channel_VisionTransformer backbone that supports Masked Autoencoder
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 collpase_embed=True,
                 **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, **kwargs) # TODO: check this
        # Additional args for Channel_ViT
        self.collpase_embed = collpase_embed
        # override the patch embedding
        self.patch_embed = PatchEmbedPerChannel(img_size, patch_size, in_chans, embed_dim)
        # TODO: check self.num_patches .etc here
        # TODO: check in Channel ViT, how is the patch embedding done?
    
    # ViT Forward path
    def forward(self, x, channel_mask_ratio=0, channel_ids=None, return_dict=False):
        x, _ = self.patch_embed(x, channel_mask_ratio, channel_ids)
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

        if not self.collpase_embed:
            raise NotImplementedError

        if return_dict:
            return dict(cls_token=cls_token, register_tokens=register_tokens, patch_tokens=patch_tokens, latent=x)
        return x, cls_token, register_tokens, patch_tokens

    # Encoder for MAE
    def forward_encoder(self, x, mask_ratio=0.75, channel_mask_ratio=0.5, channel_ids=None, return_dict=False):
        # embed patches
        x, channel_mask = self.patch_embed(x, channel_mask_ratio, channel_ids)
        channel_mask = channel_mask.unsqueeze(1).expand(-1, self.patch_size**2, -1).flatten(1)
        # if self.collpase_embed:
        #     x = collpase_channels(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

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

        if not self.collpase_embed:
            raise NotImplementedError

        if return_dict:
            return dict(latent=x, mask=mask, ids_restore=ids_restore, channel_mask=channel_mask)
        return x, mask, ids_restore, channel_mask

        