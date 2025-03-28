from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.layers import Mlp, DropPath, use_fused_attn
from itertools import product
import math

__all__ = ['AttentionPool', 'AttentionBranch', 'LowRankAttention', 'LayerScale', 'LowRankBlock']

def get_perception_field_mask(num_patches, patch_size, spacial_resolution, attention_radius, cls_token=False):
    """
    Create a distance mask for the image.
    spacial_resolution: the resolution of the image in meters/pixel
    attention_radius: the radius of the attention in meters
    """
    points = list(product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))
    idxs = []
    points_array = torch.tensor(points).to(torch.float32)
    distances = torch.cdist(points_array, points_array, p=2) * patch_size * spacial_resolution
    idxs = (distances <= attention_radius).flatten().tolist()
    mask = torch.tensor(idxs).to(torch.float32).reshape(num_patches, num_patches)
    if cls_token:   
        # add a new row and column of ones to the mask for cls_token
        new_row = torch.ones(num_patches, 1)
        new_col = torch.ones(1, num_patches + 1)
        mask = torch.cat([new_row, mask], dim=1)
        mask = torch.cat([new_col, mask], dim=0)
    # convert to boolean
    mask = mask > 0 # HW HW
    return mask

class AvgPool(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape B C N D
            mask (torch.Tensor): mask tensor of shape B N
        """
        B, C, N, D = x.shape
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1).expand(B, C, N, 1)
            x = (x * mask).sum(dim=2) / mask.sum(dim=2)
        else:
            x = x.mean(dim=2)
        x = self.linear(x)
        return x # B C D

class ProjectionPool(nn.Module):
    def __init__(self, dim: int, dim_out: int, norm_layer: nn.Module = nn.LayerNorm, **kwargs):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=True)
        self.norm = norm_layer(dim_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x is B C N D
        x = x[:, :, 0] # B C D
        x = self.linear(x)
        x = self.norm(x)
        return x # B C D

class AttentionPool(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        proj_drop: float = 0.,
    ) -> None:
        """
        AttentionPool module for low-rank attention in vision transformers.

        This module implements a pooling mechanism using multi-head attention.
        It processes input tensors by applying attention across the feature dimension,
        resulting in a lower-dimensional output. This is particularly useful for
        reducing the computational complexity in vision transformer architectures.

        Args:
            dim (int): Input dimension.
            dim_out (int): Output dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add bias to query, key, value projections.
            qk_norm (bool): If True, apply normalization to query and key.
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for output.
            norm_layer (nn.Module): Normalization layer to use.

        This method will use the cls token to pool the feature.
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.dim_out = dim
        self.scale = (dim // num_heads) ** -0.5  # Scaling factor for attention scores
        self.fused_attn = use_fused_attn()
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, D = x.shape
            
        x = x.reshape(B * C, N, D)  # B*C, N+1, D
        x_cls = x[:, :1, :] # B*C, 1, D

        q = self.q(x_cls).reshape(B * C, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B*C, num_heads, 1, head_dim
        kv = self.kv(x).reshape(B * C, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 2, B*C, num_heads, N, head_dim
        k, v = kv.unbind(0) # B*C, num_heads, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ) # B*C, num_heads, 1, D
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale # B*C, num_heads, N, 1
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2) # B*C, num_heads, 1, D
            
        x = x.reshape(B, C, self.dim) # B C dim, take only cls token
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LowDimPool(nn.Module):
    def __init__(
        self,
        dim: int,
        channel_dim: int,
        spatial_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        skip_pool: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        
        self.channel_norm = norm_layer(channel_dim)
        self.spatial_norm = norm_layer(spatial_dim)
        # self.norm = norm_layer(dim)
        
        if skip_pool:
            self.channel_pool = lambda x: x[:, :, 0]
            self.spatial_pool = lambda x: x[:, :, 0]
        else:
            self.channel_pool = AttentionPool(dim=channel_dim, num_heads=num_heads, norm_layer=norm_layer, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop) # B C N D -> B C dim
            self.spatial_pool = AttentionPool(dim=spatial_dim, num_heads=num_heads, norm_layer=norm_layer, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop) # B HW N D -> B HW dim
        
        self.channel_linear = nn.Linear(dim, channel_dim) # B C dim -> B C channel_dim
        self.spatial_linear = nn.Linear(dim, spatial_dim) # B HW dim -> B HW spatial_dim
        
    def forward(self, x: torch.Tensor, pos_chan_embedding: torch.Tensor = None) -> torch.Tensor:
        # x is B C HW D
        B, C, HW, D = x.shape
        x_c = self.channel_linear(x) # B, C, HW, channel_dim
        x_s = self.spatial_linear(x) # B, C, HW, spatial_dim
        
        x_c = x_c[:, :, 0] + self.channel_pool(self.channel_norm(x_c)) # B, C, dim
        x_s = x_s[:, 0] + self.spatial_pool(self.spatial_norm(x_s.transpose(1, 2))) # B, HW, dim
        
        # xs = x_s.reshape(B, HW, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, HW, spatial_dim
        # xc = x_c.reshape(B, C, self.num_heads, -1).permute(0, 2, 1, 3) # B, num_heads, C, channel_dim
        
        # x = torch.einsum('...ca,...nb->...cnab', xc, xs).flatten(-2) # B, num_heads, C, HW, D
        # x = x.permute(0, 2, 3, 1, 4).reshape(B, C, HW, D)
        # if pos_chan_embedding is not None:
        #     x = x + pos_chan_embedding
        # x = self.norm(x)
        return x_c, x_s, x

class AttentionBranch(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            rank: int = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn() 

        self.qk = nn.Linear(dim, num_heads * head_dim * rank * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, num_heads * head_dim, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.rank = rank
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, num_heads, N, D
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.rank, self.head_dim).permute(2, 0, 4, 3, 1, 5) # 2, B, rank, num_heads, N, head_dim
        q, k = qk.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) #B, rank, num_heads, N, head_dim
        v = v.unsqueeze(1).expand(-1, self.rank, -1, -1, -1) # B, rank, num_heads, N, D
    
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ) # B, rank, num_heads, N, D
        else:
            L, S = q.shape[-2], k.shape[-2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                atten_bias = torch.zeros(L, S, device=attn.device)
                atten_bias = atten_bias.masked_fill(mask == 0, -torch.inf)
                attn = attn + atten_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # B, rank, num_heads, N, D

        return x # B, rank, num_heads, N, D

class LowRankAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            channel_dim: int,
            spatial_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            rank: int = 1,
    ) -> None:
        """
        LowRankAttention module for low-rank attention in vision transformers.

        This module implements a low-rank attention mechanism using multi-head attention.
        It processes input tensors by applying attention across the feature dimension,
        resulting in a lower-dimensional output. This is particularly useful for
        reducing the computational complexity in vision transformer architectures.

        Args:
            dim (int): Input dimension. 
            channel_dim (int): Channel dimension.
            spatial_dim (int): Spatial dimension.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add bias to query, key, value projections.
            qk_norm (bool): If True, apply normalization to query and key.
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for output.
            norm_layer (nn.Module): Normalization layer to use.
            dim_ratio (float): Dimension ratio for low-rank approximation.
            pool (bool): If True, use pooling.
        """
        super().__init__()
        assert channel_dim % num_heads == 0, 'channel_dim should be divisible by num_heads'
        assert spatial_dim % num_heads == 0, 'spatial_dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.dim = dim
        self.channel_dim = channel_dim
        self.spatial_dim = spatial_dim
        
        self.c_head_dim = self.channel_dim // num_heads
        self.s_head_dim = self.spatial_dim // num_heads
        self.head_dim = self.dim // num_heads
        
        self.rank = rank
        
        assert self.head_dim == self.c_head_dim * self.s_head_dim, 'head_dim should be equal to c_head_dim * s_head_dim'

        self.channel_branch = AttentionBranch(channel_dim, num_heads, self.c_head_dim, qkv_bias, qk_norm, attn_drop, norm_layer, rank)
        self.spatial_branch = AttentionBranch(spatial_dim, num_heads, self.s_head_dim, qkv_bias, qk_norm, attn_drop, norm_layer, rank)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_c: torch.Tensor, x_s: torch.Tensor, spatial_mask: torch.Tensor = None) -> torch.Tensor:
        B, C, D = x_c.shape
        HW = x_s.shape[1]
        
        xc = self.channel_branch(x_c)  # B, rank, num_heads, C, c_head_dim
        xs = self.spatial_branch(x_s, spatial_mask)  # B, rank, num_heads, HW, s_head_dim
        x = torch.einsum('...ca,...nb->...cnab', xc, xs).flatten(-2) # B, rank, num_heads, C, HW, D
        x = x.sum(dim=1) # B, num_heads, C, HW, D
        x = x.permute(0, 2, 3, 1, 4).reshape(B, C, HW, -1) # B, C, HW, D
        x = self.proj(x) # B, C, HW, D
        x = self.proj_drop(x) # B, C, HW, D
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class LowRankBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            channel_dim: int,
            spatial_dim: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            skip_pool: bool = False,
            rank: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.spatial_norm = norm_layer(spatial_dim)
        self.channel_norm = norm_layer(channel_dim)

        self.low_dim_pool = LowDimPool(
            dim=dim,
            channel_dim=channel_dim,
            spatial_dim=spatial_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            skip_pool=skip_pool,
        )
        
        self.attn = LowRankAttention(
            dim=dim,
            num_heads=num_heads,
            channel_dim=channel_dim,
            spatial_dim=spatial_dim,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rank=rank,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor = None, pos_chan_embedding: torch.Tensor = None) -> torch.Tensor:
        x_c, x_s, _ = self.low_dim_pool(self.norm1(x), pos_chan_embedding)
        x = x + self.drop_path1(self.ls1(self.attn(self.channel_norm(x_c), self.spatial_norm(x_s), spatial_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
