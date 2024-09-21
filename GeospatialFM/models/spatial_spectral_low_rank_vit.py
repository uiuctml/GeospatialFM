import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Dict, Any, List
from functools import partial
import numpy as np
from .hyperspectral_patch_embed import  HyperspectralPatchEmbed
from .pos_chan_embed import PositionalChannelEmbedding
from .low_rank_attention import LowRankBlock, get_perception_field_mask

class SpatialSpectralLowRankViTConfig(PretrainedConfig):
    model_type = "multi_modal_low_rank_vit"

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        channel_embed_dims_per_head: int = 4,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # Decoder-specific parameters
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_channel_embed_dims_per_head: int = 4,
        decoder_num_heads: int = 16,
        decoder_out_chans: int = 1,
        # return dict
        return_dict: bool = False,
        norm_pix_loss: bool = True,
        use_perception_field_mask: bool = False,
        attention_radius: int = 640,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channel_dim = channel_embed_dims_per_head * num_heads
        self.spatial_dim = embed_dim // self.channel_dim * num_heads  
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_path_rate = drop_path_rate
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.num_tokens = 1
        self.return_dict = return_dict

        # Decoder-specific attributes
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_channel_dim = decoder_channel_embed_dims_per_head * decoder_num_heads
        self.decoder_spatial_dim = decoder_embed_dim // self.decoder_channel_dim * decoder_num_heads
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_out_chans = decoder_out_chans
        
        # MAE-specific attributes
        self.norm_pix_loss = norm_pix_loss
        
        # Perception field mask
        self.use_perception_field_mask = use_perception_field_mask
        self.attention_radius = attention_radius

    @property
    def encoder_config(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('decoder_')}

    @property
    def decoder_config(self):
        return {
            'embed_dim': self.decoder_embed_dim,
            'depth': self.decoder_depth,
            'num_heads': self.decoder_num_heads,
            'out_chans': self.decoder_out_chans,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_norm': self.qk_norm,
            'drop_path_rate': self.drop_path_rate,
            'drop_path_uniform': self.drop_path_uniform,
            'init_values': self.init_values,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop,
            'channel_dim': self.channel_dim,
            'spatial_dim': self.spatial_dim,
            'patch_size': self.patch_size,
            'return_dict': self.return_dict,
        }

class SpatialSpectralLowRankViTEncoder(PreTrainedModel):
    config_class = SpatialSpectralLowRankViTConfig

    def __init__(self, config: SpatialSpectralLowRankViTConfig):
        super().__init__(config)
        self.config = config
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # Patch embedding layers for optical and radar inputs
        self.optical_patch_embed = HyperspectralPatchEmbed(config.patch_size, config.embed_dim)
        self.radar_patch_embed = HyperspectralPatchEmbed(config.patch_size, config.embed_dim)
        
        # Learnable tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_chan_embed = PositionalChannelEmbedding(config.embed_dim)
        
        if config.drop_path_uniform is True:
            dpr = [config.drop_path_rate] * config.depth
        else:
            dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            LowRankBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                channel_dim=config.channel_dim,
                spatial_dim=config.spatial_dim,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_norm=config.qk_norm,
                proj_drop=config.proj_drop,
                attn_drop=config.attn_drop,
                drop_path=dpr[i],
                init_values=config.init_values,
                norm_layer=norm_layer,
            )
            for i in range(config.depth)
        ])
        
        # Final normalization layer
        self.norm = norm_layer(config.embed_dim)
        self.head = nn.Identity()
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        w = self.optical_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10, mask_ratio=0, channel_mask_ratio=0):
        assert optical is not None or radar is not None, "At least one of optical and radar should be provided"
        assert optical_channel_wv is None or len(optical_channel_wv.shape) == 2, "If optical ids are provided, they should be a 2D tensor"
        assert radar_channel_wv is None or len(radar_channel_wv.shape) == 2, "If radar ids are provided, they should be a 2D tensor"
        
        dummy_loss = 0 # dummy loss to avoid empty loss error
        if optical is not None:
            optical = self.optical_patch_embed(optical)  # B, Co, HW, D
            assert optical_channel_wv is not None, "Optical ids should be provided"
            assert optical_channel_wv.shape[1] == optical.shape[1], "Optical ids should have the same number of channels as the optical data"
        else:
            optical_channel_wv = None
            for param in self.optical_patch_embed.parameters():
                dummy_loss += 0.0 * torch.sum(param)
        if radar is not None:
            radar = self.radar_patch_embed(radar)  # B, Cr, HW, D
            assert radar_channel_wv is not None, "Radar ids should be provided"
            assert radar_channel_wv.shape[1] == radar.shape[1], "Radar ids should have the same number of channels as the radar data"
        else:
            radar_channel_wv = None
            for param in self.radar_patch_embed.parameters():
                dummy_loss += 0.0 * torch.sum(param)

        channel_ids = self.maybe_concat(optical_channel_wv, radar_channel_wv) # 1, C = Co + Cr    
        x = self.maybe_concat(optical, radar) # B, C, HW, D
        num_patches = x.shape[2]
        
        # Create the perception field mask to simulate convolutional attention
        if self.config.use_perception_field_mask:
            perception_field_mask = get_perception_field_mask(num_patches, self.config.patch_size, spatial_resolution, attention_radius=self.config.attention_radius, cls_token=False).to(x.device)
        else:
            perception_field_mask = None
        
        # Add positional and channel embedding
        pos_chan_embed = self.pos_chan_embed(x, channel_ids=channel_ids, spatial_resolution=spatial_resolution, cls_token=False).to(x.device, dtype=x.dtype)
        x = x + pos_chan_embed
        
        # Apply masks after positional embedding
        if self.training:
            x, channel_mask, channel_ids_restore = self.random_channel_masking(x, channel_mask_ratio) # B N HW D, B C, B C
            x, pos_mask, pos_ids_restore, perception_field_mask = self.random_pos_masking(x, mask_ratio, perception_field_mask=perception_field_mask) # B N L D, B HW, B HW, L L
        else:
            channel_mask = pos_mask = channel_ids_restore = pos_ids_restore = None
        
        B, N, L, D = x.shape
        
        # Create cls tokens for both Cin and HW dimensions
        cls_token_cin = self.cls_token.expand(B, 1, L, D)
        cls_token_hw = self.cls_token.expand(B, N + 1, 1, D)
        
        # Append cls token to Cin dimension
        x = torch.cat((cls_token_cin, x), dim=1)
        
        # Append cls token to HW dimension
        x = torch.cat((cls_token_hw, x), dim=2)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, spatial_mask=perception_field_mask)

        # Apply final layer norm
        x = self.norm(x)  # B N+1 L+1 D
        
        cls_token = x[:, 0, 0]
        patch_tokens = x[:, 1:, 1:] # B N+1 L+1 D -> B N L D

        if self.training:
            return x + dummy_loss, channel_mask, channel_ids_restore, pos_mask, pos_ids_restore
        elif self.config.return_dict:
            return BaseModelOutput(
                last_hidden_state=cls_token,
                hidden_states=patch_tokens,
                attentions=None,
            )
        else:
            return x, cls_token, patch_tokens

    def maybe_concat(self, x, y):
        if x is not None and y is not None:
            return torch.cat((x, y), dim=1)
        elif x is not None:
            return x
        elif y is not None:
            return y
        
    def random_pos_masking(self, x, mask_ratio, batch_wise_mask=False, perception_field_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, C, HW, D], sequence
        perception_field_mask: [HW, HW] or None, if not None, use distance attention mask is used.
        """
        if perception_field_mask is not None:
            batch_wise_mask = True # if distance mask is provided, we need to use batch-wise masking
            
        B, C, HW, D = x.shape  # batch, channels, length, dim
        if mask_ratio == 0:
            mask = torch.zeros([B, HW], device=x.device)
            ids_restore = torch.arange(HW, device=x.device).unsqueeze(0).repeat(B, 1)
            return x, mask, ids_restore
        
        x = x.permute(0, 2, 1, 3) # B, HW, C, D
        len_keep = int(HW * (1 - mask_ratio)) # L
        
        if batch_wise_mask:
            noise = torch.rand(1, HW, device=x.device)  # noise in [0, 1]
            noise = noise.repeat(B, 1)
        else:
            noise = torch.rand(B, HW, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # B, HW

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # B, L
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, C, D)) # B, L, C, D
        if perception_field_mask is not None:
            # remove the corresponding ids in perception_field_mask
            ids_keep_ = ids_keep[0].cpu() # L
            perception_field_mask = perception_field_mask[ids_keep_, :] # L HW
            perception_field_mask = perception_field_mask[:, ids_keep_] # L L
            # append cls token
            new_row = torch.ones(len_keep, 1, device=perception_field_mask.device)
            new_col = torch.ones(1, len_keep + 1, device=perception_field_mask.device)
            perception_field_mask = torch.cat([new_row, perception_field_mask], dim=1)
            perception_field_mask = torch.cat([new_col, perception_field_mask], dim=0)
            perception_field_mask = perception_field_mask > 0 # L L bool

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, HW], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # B, HW
        
        x_masked = x_masked.permute(0, 2, 1, 3)  # [B, C, L, D]
        return x_masked, mask, ids_restore, perception_field_mask
    
    def random_channel_masking(self, x, channel_mask_ratio, batch_wise_mask=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, C, HW, D], sequence
        Return:
            x_masked: [B, N, HW, D]
            mask: [B, C]
            ids_restore: [B, C]
        """
        B, C, HW, D = x.shape
        if channel_mask_ratio == 0:
            mask = torch.zeros([B, C], device=x.device)
            ids_restore = torch.arange(C, device=x.device).unsqueeze(0).repeat(B, 1)
            return x, mask, ids_restore
        
        len_keep = max(int(C * (1 - channel_mask_ratio)), 2) # at least 2 channel is kept, N
        
        if batch_wise_mask:
            noise = torch.rand(1, C, device=x.device)  # noise in [0, 1]
            noise = noise.repeat(B, 1)
        else:
            noise = torch.rand(B, C, device=x.device) # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1) # B C
        ids_restore = torch.argsort(ids_shuffle, dim=1) # B C

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, HW, D)) # B, N, HW, D

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, C], device=x.device) # B C
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # B C
    
        return x_masked, mask, ids_restore

class SpatialSpectralLowRankViTDecoder(PreTrainedModel):
    config_class = SpatialSpectralLowRankViTConfig

    def __init__(self, config: SpatialSpectralLowRankViTConfig):
        super().__init__(config)
        self.config = config
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.decoder_embed_dim))
        
        if config.drop_path_uniform is True:
            dpr = [config.drop_path_rate] * config.decoder_depth
        else:
            dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.decoder_depth)]
        
        self.decoder_blocks = nn.ModuleList([
            LowRankBlock(
                dim=config.decoder_embed_dim,
                num_heads=config.decoder_num_heads,
                channel_dim=config.decoder_channel_dim,
                spatial_dim=config.decoder_spatial_dim,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_norm=config.qk_norm,
                proj_drop=config.proj_drop,
                attn_drop=config.attn_drop,
                drop_path=dpr[i],
                init_values=config.init_values,
                norm_layer=norm_layer,
            )
            for i in range(config.decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(config.decoder_embed_dim, config.patch_size**2, bias=True)
        
        self.pos_chan_embed = PositionalChannelEmbedding(config.decoder_embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos_ids_restore, channel_ids_restore, optical_channel_wv, radar_channel_wv, spatial_resolution, restore_input_dim=False):
        assert len(optical_channel_wv.shape) == 2, "Optical ids should be a 2D tensor"
        assert len(radar_channel_wv.shape) == 2, "Radar ids should be a 2D tensor"
        
        # embed tokens
        x = self.decoder_embed(x)  # B N+1 L+1 D
        
        channel_wv = torch.cat((optical_channel_wv, radar_channel_wv), dim=1)  # 1 C = Co + Cr or B C
        
        # remove cls token
        B, N, L, D = x[:, 1:, 1:, :].shape
        C = channel_ids_restore.shape[1]
        HW = pos_ids_restore.shape[1]
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.expand(B, N + 1, HW - L, -1)
        x_ = torch.cat([x[:, :, 1:], mask_tokens], dim=2)  # B N+1 HW D, remove cls token
        x_ = torch.gather(x_, dim=2, index=pos_ids_restore.unsqueeze(1).unsqueeze(-1).repeat(1, N+1, 1, D))  # unshuffle, B N+1 HW D
        x = torch.cat([x[:, :, :1], x_], dim=2)  # B N+1 HW+1 D, add cls token
        
        # append mask tokens for channels
        mask_tokens = self.mask_token.expand(B, C - N, HW + 1, -1)
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)  # B, C, HW+1, D, remove cls token
        x_ = torch.gather(x_, dim=1, index=channel_ids_restore.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, HW + 1, D))  # unshuffle, B C HW+1 D
        x = torch.cat([x[:, :1], x_], dim=1)  # B C+1 HW+1 D, add cls token
        
        if C == optical_channel_wv.shape[1]: 
            # missing radar channels from encoder
            n_radar_channels = radar_channel_wv.shape[1]
            # extend x to C+n_radar_channels to the end
            mask_tokens = self.mask_token.expand(B, n_radar_channels, HW + 1, -1)  # B C HW+1 D
            x = torch.cat([x, mask_tokens], dim=1)
        elif C == radar_channel_wv.shape[1]:
            # missing optical channels from encoder
            n_optical_channels = optical_channel_wv.shape[1]
            # extend x to C+n_optical_channels to the front behind the cls token
            mask_tokens = self.mask_token.expand(B, n_optical_channels, HW + 1, -1)
            x = torch.cat([x[:, :1, :, :], mask_tokens, x[:, 1:, :, :]], dim=1)  # B C+1 HW+1 D

        # add positional and channel embedding
        pos_chan_embed = self.pos_chan_embed(x[:, 1:, 1:, :], channel_ids=channel_wv, spatial_resolution=spatial_resolution, cls_token=True).to(x.device, dtype=x.dtype)
        x = x + pos_chan_embed

        num_patches = x.shape[2] - 1
        if self.config.use_perception_field_mask:
            perception_field_mask = get_perception_field_mask(num_patches, self.config.patch_size, spatial_resolution, attention_radius=self.config.attention_radius, cls_token=True).to(x.device)
        else:
            perception_field_mask = None

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, spatial_mask=perception_field_mask)

        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)  # B C+1 HW+1 D -> B C+1 HW+1 patch_size**2
        
        # remove cls token
        x = x[:, 1:, 1:, :]  # B C HW patch_size**2
        
        if restore_input_dim:
            x = self.unpatchify(x)
        
        if self.config.return_dict:
            # Return as BaseModelOutput for Hugging Face compatibility
            return BaseModelOutput(
                last_hidden_state=x,
                hidden_states=None,
                attentions=None,  # You can modify this if you have attention weights to return
            )
        else:
            return x

    def unpatchify(self, x):
        """
        x: (B, C, HW, patch_size**2)
        imgs: (B, C, H, W)
        """
        p = self.config.patch_size
        h = w = int(x.shape[2]**.5)
        assert h * w == x.shape[2]
        
        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p))
        x = torch.einsum('bchwpq->bchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], h * p, h * p))
        return imgs

    def forward_target(self, imgs):
        """
        imgs: [B, C, H, W]
        """
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        return target # B C HW patch_size**2

    def patchify(self, imgs):
        """
        imgs: (B, C, H, W)
        x: (B, C, HW, patch_size**2)
        """
        p = self.config.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('bchpwq->bchwpq', x)
        x = x.reshape(shape=(imgs.shape[0], imgs.shape[1], h * w, p**2))
        return x
