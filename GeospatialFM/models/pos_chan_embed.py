# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

from typing import Any
import numpy as np

import math
import torch
import torch.nn as nn
from torch import Tensor, nn
from typing import Literal


class PositionalChannelEmbedding():
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.spactial_resolution = None
        self.channel_ids = None
        self.num_patches = None
        self.pos_embed = None
        self.channel_embed = None
        self.cls_token = torch.zeros(1, 1, 1, embed_dim)
        

    def interpolate_pos_channel_embed(self, pos_embed: torch.Tensor, channel_embed: torch.Tensor, cls_token: bool = False):
        """
        pos_embed: (1, HW, D)
        channel_embed: (1, C, D)
        return pos_embed: (1, C+1, HW+1, D) if cls_token else (1, C, HW, D)
        """
        n_chan = channel_embed.shape[1] # C
        n_pos = pos_embed.shape[1] # HW
        
        if cls_token:
            # Add cls token embeddings
            pos_cls_embed = torch.zeros(1, 1, self.embed_dim, device=pos_embed.device)
            pos_embed = torch.cat([pos_cls_embed, pos_embed], dim=1)  # (1, HW+1, D)
            
            chan_cls_embed = torch.zeros(1, 1, self.embed_dim, device=channel_embed.device)
            channel_embed = torch.cat([chan_cls_embed, channel_embed], dim=1)  # (1, C+1, D)
            
            n_chan += 1
            n_pos += 1
            
        interpolated_pos_embed = pos_embed.unsqueeze(1).repeat(1, n_chan, 1, 1) # 1 C HW D
        interpolated_channel_embed = channel_embed.unsqueeze(2).repeat(1, 1, n_pos, 1) # 1 C HW D
        
        pos_channel_embed = (interpolated_channel_embed + interpolated_pos_embed) / 2 # 1 C HW D
        
        return pos_channel_embed
    
    def get_pos_embed(self, tokens: torch.Tensor, spatial_resolution: float, cls_token: bool = True):
        _, HW, _ = tokens.shape
        grid_size = int(np.sqrt(HW))
        assert grid_size * grid_size == HW, "HW must be a square"
        
        if self.spactial_resolution == spatial_resolution and self.num_patches == HW:
            pos_embed = self.pos_embed
        else:
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size, spatial_resolution, cls_token=False) # (1, HW, D)
            self.pos_embed = pos_embed
            self.spactial_resolution = spatial_resolution
            self.num_patches = HW
            
        if cls_token:
            pos_embed = torch.cat([torch.zeros_like(pos_embed[:, :1, :]), pos_embed], dim=1)
            
        return pos_embed
    
    def get_channel_embed(self, tokens: torch.Tensor, channel_ids: torch.Tensor, cls_token: bool = True):
        """
        channel_ids: (1, C)
        return channel_embed: (1, C, D)
        """
        _, C, _ = tokens.shape
        channel_ids = channel_ids.squeeze(0).detach().cpu().numpy()
        if self.channel_ids is not None and tuple(self.channel_ids) == tuple(channel_ids):
            channel_embed = self.channel_embed
        else:
            channel_embed = get_1d_sincos_channel_embed(self.embed_dim, channel_ids, cls_token=False) # (1, C, D)
            self.channel_embed = channel_embed
            self.channel_ids = channel_ids
        
        if cls_token:
            channel_embed = torch.cat([torch.zeros_like(channel_embed[:, :1, :]), channel_embed], dim=1)
        return channel_embed
        
    
    def __call__(self, tokens: torch.Tensor, spatial_resolution: float, channel_ids: torch.Tensor, cls_token: bool = True):
        _, C, HW, _ = tokens.shape
        assert channel_ids.shape == (1, C), "channel_ids must be the same length as the number of channels"
        grid_size = int(np.sqrt(HW))
        assert grid_size * grid_size == HW, "HW must be a square"
        channel_ids = channel_ids.squeeze(0).detach().cpu().numpy()
        
        if self.channel_ids is not None and tuple(self.channel_ids) == tuple(channel_ids):
            channel_embed = self.channel_embed
        else:
            channel_embed = get_1d_sincos_channel_embed(self.embed_dim, channel_ids, cls_token=False) # (1, C, D)
            self.channel_embed = channel_embed
            self.channel_ids = channel_ids
            
        if self.spactial_resolution == spatial_resolution and self.num_patches == HW:
            pos_embed = self.pos_embed
        else:
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size, spatial_resolution, cls_token=False) # (1, HW, D)
            self.pos_embed = pos_embed
            self.spactial_resolution = spatial_resolution
            self.num_patches = HW
            
        return self.interpolate_pos_channel_embed(pos_embed, channel_embed, cls_token)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, spactial_resolution, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0) * spactial_resolution

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
    return pos_embed

def get_1d_sincos_channel_embed(embed_dim, channel_idx, cls_token=False):
    """
    embed_dim: output dimension for each position
    channel_idx: a list of channel_idx to be encoded: size (C,)
    out: (C, D)
    """
    channel_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, channel_idx)
    if cls_token:
        channel_embed = np.concatenate([np.zeros([1, embed_dim]), channel_embed], axis=0)
    channel_embed = torch.from_numpy(channel_embed).float().unsqueeze(0)
    return channel_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
# --------------------------------------------------------
# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionChannelEmbedding(nn.Module):
    def __init__(
        self,
        # embed_dim: int,
        spatial_dim: int,
        channel_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert spatial_dim % (4 * num_heads) == 0 and spatial_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head_hw = spatial_dim // num_heads
        D_head_c = channel_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head_hw = D_head_hw
        self.D_head_c = D_head_c
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods_hw",
            torch.empty(D_head_hw // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "periods_c",
            torch.empty(D_head_c // 2, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int, C: int) -> list[Tensor, ...]:
        device = self.periods_hw.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords_hw = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords_hw = coords_hw.flatten(0, 1)  # [HW, 2]
        coords_hw = 2.0 * coords_hw - 1.0  # Shift range [0, 1] to [-1, +1]

        coords_c = torch.arange(0.5, C, **dd) / C # [C]
        coords_c = 2.0 * coords_c - 1.0

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords_hw += shift_hw[None, :]
            
            shift_c = torch.empty(1, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords_c += shift_c

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords_hw *= jitter_hw[None, :]

            jitter_c = torch.empty(1, **dd).uniform_(jitter_min, jitter_max).exp()
            coords_c *= jitter_c

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords_hw *= rescale_hw
            coords_c *= rescale_c

        # Prepare angles and sin/cos
        angles_hw = 2 * math.pi * coords_hw[:, :, None] / self.periods_hw[None, None, :]  # [HW, 2, D//4]
        angles_hw = angles_hw.flatten(1, 2)  # [HW, D//2]
        angles_hw = angles_hw.tile(2)  # [HW, D]
        cos_hw = torch.cos(angles_hw)  # [HW, D]
        sin_hw = torch.sin(angles_hw)  # [HW, D]

        angles_c = 2 * math.pi * coords_c[:, None] / self.periods_c[None, :]  # [C, D//2]
        angles_c = angles_c.tile(2)  # [C, D]
        cos_c = torch.cos(angles_c)  # [C, D]
        sin_c = torch.sin(angles_c)  # [C, D]

        # rope = {
        #     'spatial_rope': (sin_hw, cos_hw), # 2 * [HW, D]
        #     'spectral_rope': (sin_c, cos_c) # 2 * [C, D]
        # }
        # return (sin, cos)  # 2 * [HW, D]
        return [sin_hw, cos_hw, sin_c, cos_c]

    def _init_weights(self):
        device = self.periods_hw.device
        dtype = self.dtype
        if self.base is not None:
            periods_hw = self.base ** (
                2 * torch.arange(self.D_head_hw // 4, device=device, dtype=dtype) / (self.D_head_hw // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head_hw // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods_hw = base**exponents  # range [1, max_period / min_period]
            periods_hw = periods_hw / base  # range [min_period / max_period, 1]
            periods_hw = periods_hw * self.max_period  # range [min_period, max_period]

        if self.base is not None:
            periods_c = self.base ** (
                2 * torch.arange(self.D_head_c // 2, device=device, dtype=dtype) / (self.D_head_c)
            )  # [D//2]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head_c // 2, device=device, dtype=dtype)
            periods_c = base**exponents
            periods_c = periods_c / base
            periods_c = periods_c * self.max_period

        self.periods_hw.data = periods_hw
        self.periods_c.data = periods_c

# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    x_ = x[:, :, :, 1:, :]
    # print(x_.shape, cos.unsqueeze(1).unsqueeze(1).shape)
    x_ = (x_ * cos.unsqueeze(1).unsqueeze(1)) + (rope_rotate_half(x_) * sin.unsqueeze(1).unsqueeze(1))
    x = torch.cat((x[:, :, :, :1, :], x_), dim=3)
    return x