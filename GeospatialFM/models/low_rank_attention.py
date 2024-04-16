from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.layers import Mlp, DropPath, use_fused_attn

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this

class LowRankAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            dim_ratio: float = 0.25,
            pool: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.c_head_dim = int(1 / dim_ratio)
        self.s_head_dim = int(self.head_dim * dim_ratio)
        assert self.c_head_dim * self.s_head_dim == self.head_dim, '1/dim_ratio should be a factor of head_dim'
        self.fused_attn = use_fused_attn()
        self.pool = pool
        if self.pool:
            print('Using NCD Pooling')

        self.qkv_c = nn.Linear(dim, int(num_heads * 3 / dim_ratio), bias=qkv_bias)
        self.qkv_s = nn.Linear(dim, int(dim * 3 * dim_ratio), bias=qkv_bias)
        self.qc_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.kc_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qs_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.ks_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def pool_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, D = x.shape
        # x_c = x.max(-3).values
        # x_s = x.max(-2).values
        x_c = x.mean(-3)
        x_s = x.mean(-2)
        qkv_c = self.qkv_c(x_c).reshape(B, C, 3, self.num_heads, self.c_head_dim).permute(2, 0, 3, 1, 4) # 3, B, num_heads, C, c_head_dim
        qkv_s = self.qkv_s(x_s).reshape(B, N, 3, self.num_heads, self.s_head_dim).permute(2, 0, 3, 1, 4) # 3, B, num_heads, N, s_head_dim
        qc, kc, vc = qkv_c.unbind(0)
        qs, ks, vs = qkv_s.unbind(0)
        qc, kc, qs, ks = self.qc_norm(qc), self.kc_norm(kc), self.qs_norm(qs), self.ks_norm(ks)
        
        if self.fused_attn:
            xs = F.scaled_dot_product_attention(
                qs, ks, vs,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ) # B, num_heads, N, s_head_dim
            xc = F.scaled_dot_product_attention(
                qc, kc, vc,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ) # B, num_heads, C, c_head_dim
            x = torch.einsum('...ca,...nb->...cnab', xc, xs).flatten(-2)
        else:
            qs = qs * self.scale
            attn_s = qs @ ks.transpose(-2, -1)
            attn_s = attn_s.softmax(dim=-1)
            attn_s = self.attn_drop(attn_s)
            xs = attn_s @ vs
            
            qc = qc * self.scale
            attn_c = qc @ kc.transpose(-2, -1)
            attn_c = attn_c.softmax(dim=-1)
            attn_c = self.attn_drop(attn_c)
            xc = attn_c @ vc
            
            x = torch.einsum('...ca,...nb->...cnab', xc, xs).flatten(-2)
        return x
           
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, D = x.shape
        qkv_c = self.qkv_c(x).reshape(B, N, C, 3, self.num_heads, self.c_head_dim).permute(3, 0, 4, 1, 2, 5) # 3, B, num_heads, N, C, c_head_dim
        qkv_s = self.qkv_s(x).reshape(B, N, C, 3, self.num_heads, self.s_head_dim).permute(3, 0, 4, 1, 2, 5) # 3, B, num_heads, N, C, s_head_dim
        qc, kc, vc = qkv_c.unbind(0)
        qs, ks, vs = qkv_s.unbind(0)      
        qc, kc, qs, ks = self.qc_norm(qc), self.kc_norm(kc), self.qs_norm(qs), self.ks_norm(ks)
        qs, ks, vs = qs.transpose(-2, -3), ks.transpose(-2, -3), vs.transpose(-2, -3)
        
        if self.fused_attn:
            xs = F.scaled_dot_product_attention(
                qs, ks, vs,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-2, -3)
            xc = F.scaled_dot_product_attention(
                qc, kc, vc,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = torch.einsum('...a,...b->...ab', xc, xs).flatten(-2)
        else:
            qs = qs * self.scale
            attn_s = qs @ ks.transpose(-2, -1)
            attn_s = attn_s.softmax(dim=-1)
            attn_s = self.attn_drop(attn_s)
            xs = attn_s @ vs
            xs = xs.transpose(-2, -3)
            
            qc = qc * self.scale
            attn_c = qc @ kc.transpose(-2, -1)
            attn_c = attn_c.softmax(dim=-1)
            attn_c = self.attn_drop(attn_c)
            xc = attn_c @ vc
            
            x = torch.einsum('...a,...b->...ab', xc, xs).flatten(-2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, 'LowRankAttention only supports 4D input'
        B, N, C, D = x.shape

        if self.pool:
            x = self.pool_forward(x)
        else:
            x = self._forward(x)
                
        x = x.transpose(1, 2).reshape(B, N, C, D)
        # if self.pool:
        #     x = x.sum(-2)
        #     assert x.shape == (B, N, D)
        #     print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
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
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            dim_ratio: float = 0.25,
            pool: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LowRankAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            dim_ratio=dim_ratio,
            pool=pool,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
