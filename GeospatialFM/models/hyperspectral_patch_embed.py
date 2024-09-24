import torch
import torch.nn as nn
import numpy as np

class HyperspectralPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        
        # self.proj = nn.Conv3d(
        #     1,
        #     embed_dim,
        #     kernel_size=(1, patch_size, patch_size),
        #     stride=(1, patch_size, patch_size),
        # )
        
        self.proj = nn.Sequential(
            nn.Conv3d(
                1,
                embed_dim*2, #2
                kernel_size=(1, patch_size, patch_size),
                stride=(1, patch_size, patch_size),
            ),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim*2,
                embed_dim,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):
        try:
            w = self.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        except:
            w1 = self.proj[0].weight.data
            w2 = self.proj[2].weight.data
            torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))

    def forward(self, x):
        # x: B, Cin, H, W
        assert x.shape[2] % self.patch_size == 0, "image height should be divisible by patch size"
        assert x.shape[3] % self.patch_size == 0, "image width should be divisible by patch size"
        x = self.proj(x.unsqueeze(1)) # B, D, C, H, W
        B, D, C, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, C, H*W, D) # B, C, H, W, D
