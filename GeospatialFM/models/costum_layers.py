import torch.nn as nn
from torch.nn.init import trunc_normal_
import random

class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        collapse_channels: bool = False,
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
        self.collapse_channels = collapse_channels

        trunc_normal_(self.channel_embed.weight, std=0.02)

    def forward(self, x, extra_tokens={}):
        # embedding lookup
        cur_channel_embed = self.channel_embed(
            extra_tokens["channels"]
        )  # B, Cin, embed_dim=Cout
        cur_channel_embed = cur_channel_embed.permute(0, 2, 1)  # B Cout Cin

        B, Cin, H, W = x.shape
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans

        if self.training and self.enable_sample:
            # Per batch channel sampling
            # Note this may be slow
            # Randomly sample the number of channels for this batch
            Cin_new = random.randint(1, Cin)

            # Randomly sample the selected channels
            channels = random.sample(range(Cin), k=Cin_new)
            Cin = Cin_new
            x = x[:, channels, :, :]

            # Update the embedding lookup
            cur_channel_embed = cur_channel_embed[:, :, channels]
            ######

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        x += cur_channel_embed.unsqueeze(-1).unsqueeze(-1)
        # x += self.channel_embed[:, :, cur_channels, :, :]  # B Cout Cin H W

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout
        if self.collapse_channels:
            x = x.sum()

        return x, Cin