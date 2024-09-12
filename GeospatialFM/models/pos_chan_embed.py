# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

from typing import Any
import numpy as np

import torch
import torch.nn as nn


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
        interpolated_pos_embed = pos_embed.unsqueeze(1).repeat(1, n_chan, 1, 1) # 1 C HW D
        
        n_pos = pos_embed.shape[1] # HW
        interpolated_channel_embed = channel_embed.unsqueeze(2).repeat(1, 1, n_pos, 1) # 1 C HW D
        
        pos_channel_embed = interpolated_channel_embed + interpolated_pos_embed # 1 C HW D
        
        if cls_token:
            pos_channel_embed = torch.cat([torch.zeros_like(pos_channel_embed[:, :1, :, :]), pos_channel_embed], dim=1)
            pos_channel_embed = torch.cat([torch.zeros_like(pos_channel_embed[:, :, :1, :]), pos_channel_embed], dim=2)
        return pos_channel_embed
    
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
