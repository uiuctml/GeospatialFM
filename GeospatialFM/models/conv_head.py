import os
from typing import Any, Optional, Dict
import timm
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# Temporarily import wandb and matplotlib to log images
import wandb
import matplotlib.pyplot as plt
import warnings
from typing import List

class MoEConvHead(nn.Module):
    """
    Mixture-of-Experts version of ConvHead.

    Expected input:
        features: [B, C, N, D]
            B = batch size
            C = number of channels / views / groups
            N = number of tokens per channel (including CLS if present upstream)
            D = embedding dim

    This module assumes the caller passes the full hidden states before flattening
    across channels, similar to your MoEUpperNet design.
    """
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        patch_size: int,
        num_experts: int,
        topk: int = 3,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.topk = topk

        self.experts = nn.ModuleList([
            ConvHead(
                embedding_size=embedding_size,
                num_classes=num_classes,
                patch_size=patch_size,
            )
            for _ in range(num_experts)
        ])

        # Gate on CLS token features: [B, C, D] -> [B, C, num_experts]
        self.gate = nn.Linear(embedding_size, num_experts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, N, D]
                Assumes token 0 is CLS, and tokens 1: are patch tokens.

        Returns:
            logits: [B, num_classes, H_out, W_out]
        """
        if features.ndim != 4:
            raise ValueError(
                f"Expected features of shape [B, C, N, D], got {features.shape}"
            )

        B, C, N, D = features.shape

        if self.topk == -1:
            topk = C
        else:
            topk = min(self.topk, C)

        # CLS tokens for gating
        cls_tokens = features[:, :, 0, :]          # [B, C, D]
        patch_tokens = features[:, :, 1:, :]       # [B, C, N-1, D]

        num_patches = patch_tokens.shape[2]
        H = W = int(math.sqrt(num_patches))
        if H * W != num_patches:
            raise ValueError(
                f"Number of patch tokens ({num_patches}) is not a perfect square."
            )

        # Gate scores over experts
        gate_score = self.gate(cls_tokens)         # [B, C, E]
        gate_score = gate_score.permute(0, 2, 1)   # [B, E, C]
        gate_prob = F.softmax(gate_score, dim=-1)  # soft selection over channels

        expert_logits = []

        for i in range(self.num_experts):
            # Select top-k channels for expert i
            topk_values, topk_indices = torch.topk(
                gate_prob[:, i, :], k=topk, dim=-1
            )  # both [B, topk]

            # Gather selected channel patch tokens
            gather_idx = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, num_patches, D
            )  # [B, topk, num_patches, D]

            topk_features = torch.gather(
                patch_tokens, dim=1, index=gather_idx
            )  # [B, topk, num_patches, D]

            # Normalize selected weights among top-k only
            topk_weights = F.softmax(topk_values, dim=-1).unsqueeze(-1).unsqueeze(-1)
            # [B, topk, 1, 1]

            fused_features = (topk_features * topk_weights).sum(dim=1)
            # [B, num_patches, D]

            # Reshape to image-like tensor for ConvHead
            fused_features = fused_features.transpose(1, 2).reshape(B, D, H, W)
            # [B, D, H, W]

            logits = self.experts[i](fused_features)
            expert_logits.append(logits)

        # Average across experts
        logits = torch.stack(expert_logits, dim=0).mean(dim=0)
        return logits


class ConvHead(nn.Module):
    def __init__(self, embedding_size: int = 384, num_classes: int = 5, patch_size: int = 4):
        super(ConvHead, self).__init__()

        # Ensure patch_size is a positive power of 2
        if not (patch_size > 0 and ((patch_size & (patch_size - 1)) == 0)):
            raise ValueError("patch_size must be a positive power of 2.")

        num_upsampling_steps = int(math.log2(patch_size))

        # Determine the initial number of filters (maximum 128 or embedding_size)
        initial_filters = 128

        # Generate the sequence of filters: 128, 64, 32, ..., down to num_classes
        filters = [initial_filters // (2 ** i) for i in range(num_upsampling_steps - 1)]
        filters.append(num_classes)  # Ensure the last layer outputs num_classes channels

        layers = []
        in_channels = embedding_size

        for i in range(num_upsampling_steps):
            out_channels = filters[i]

            # Upsampling layer
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

            # Convolutional layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            # Apply BatchNorm and ReLU only if not the last layer
            if i < num_upsampling_steps - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels  # Update in_channels for the next iteration

        self.segmentation_conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.segmentation_conv(x)