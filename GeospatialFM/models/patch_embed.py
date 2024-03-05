import torch
import torch.nn as nn
import numpy as np
from .pos_embed import ContinuousChannelEmbedding
from torch.nn.init import trunc_normal_

SENTINEL_WV = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4]

class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        continuous_channels: bool = True,
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

        if continuous_channels:
            self.channel_embed = ContinuousChannelEmbedding(in_chans, embed_dim)  # dynamic but fixed sin-cos embedding
        else:
            self.channel_embed = nn.Embedding(in_chans, embed_dim)
            trunc_normal_(self.channel_embed.weight, std=0.02)

        self.enable_sample = enable_sample

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

    def forward(self, x, channel_mask_ratio=0.5, channel_ids=None):
        # embedding lookup
        if channel_ids is None:
            channel_ids = np.arange(x.shape[1])
        if isinstance(self.channel_embed, nn.Embedding):
            channel_ids = torch.tensor(channel_ids, device=x.device).unsqueeze(0)#.repeat(x.shape[0], 1)
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
