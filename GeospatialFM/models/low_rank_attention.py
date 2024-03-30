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
            seperate_v: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn() and seperate_v
        self.seperate_v = not use_fused_attn() or seperate_v
        self.nqkv = 6 if self.seperate_v else 5

        self.qkv = nn.Linear(dim, dim * self.nqkv, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, 'LowRankAttention only supports 4D input'
        B, N, C, D = x.shape
        qkv = self.qkv(x).reshape(B, N, C, self.nqkv, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        if self.seperate_v:
            qc, kc, qs, ks, vc, vs = qkv.unbind(0)
            vs = vs.transpose(-2, -3)
        else:
            qc, kc, qs, ks, v = qkv.unbind(0)
            v = v.flatten(-3, -2)
        qc, kc, qs, ks = self.q_norm(qc), self.k_norm(kc), self.q_norm(qs), self.k_norm(ks)
        qs, ks = qs.transpose(-2, -3), ks.transpose(-2, -3)
        
        if self.fused_attn:
            xs = F.scaled_dot_product_attention(
                qs, ks, vs,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-2, -3)
            xc = F.scaled_dot_product_attention(
                qc, kc, vc,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = xs * xc
        else:
            qs = qs * self.scale
            attn_s = qs @ ks.transpose(-2, -1)
            attn_s = attn_s.softmax(dim=-1)
            attn_s = self.attn_drop(attn_s)
            attn_s = attn_s.transpose(-2, -3)
            
            qc = qc * self.scale
            attn_c = qc @ kc.transpose(-2, -1)
            attn_c = attn_c.softmax(dim=-1)
            attn_c = self.attn_drop(attn_c)
            
            attn = attn_c.unsqueeze(-1) @ attn_s.unsqueeze(-2)
            attn = attn.flatten(-4, -3).flatten(-2)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C, D)
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
            seperate_v: bool = True,
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
            seperate_v=seperate_v,
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
