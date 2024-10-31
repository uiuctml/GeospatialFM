from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .UPerNet import UPerNet
from transformers import PretrainedConfig
from .spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder

class LESSViTEncoderConfig(PretrainedConfig):
    model_type = "less_vit_encoder"

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
        mask_ratio: float = 0.75,
        channel_mask_ratio: float = 0.5,
        pos_chan_embed_residual: bool = True,
        return_dict: bool = False,
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
        self.mask_ratio = mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.pretrain = False
        
        # Perception field mask
        self.use_perception_field_mask = use_perception_field_mask
        self.attention_radius = attention_radius
        
        # Positional channel embedding residual
        self.pos_chan_embed_residual = pos_chan_embed_residual

class LESSWithProjectionConfig(LESSViTEncoderConfig):
    model_type = "less_with_projection"
    
    def __init__(self, num_labels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        
class LESSWithUPerNetConfig(LESSViTEncoderConfig):
    model_type = "less_with_uper_net"
    
    def __init__(self, num_labels: int, image_size: int, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.image_size = image_size

class LESSWithProjection(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = SpatialSpectralLowRankViTEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self, optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10,
    ) -> Union[Tuple, dict]:
        
        # Get encoder outputsp
        outputs = self.encoder(optical, radar, optical_channel_wv, radar_channel_wv, spatial_resolution)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        else:
            outputs = outputs.last_hidden_state
            
        # Use the [CLS] token
        pooled_output = outputs[:, 0, 0]
        
        # Get logits
        logits = self.classifier(pooled_output)

        return {"logits": logits} if self.config.return_dict else logits


class LESSWithUPerNet(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = SpatialSpectralLowRankViTEncoder(config)
        self.decoder = UPerNet(
            num_classes=config.num_labels,
            image_size=config.image_size,
            debug=False
        )

    def forward(
        self, 
        optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10,
    ) -> Union[Tuple, dict]:
        # Get encoder outputs
        outputs = self.encoder(optical, radar, optical_channel_wv, radar_channel_wv, spatial_resolution)
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state
            
        # Get segmentation logits
        logits = self.decoder(hidden_states[:, 0, 1:])

        return {"logits": logits} if self.config.return_dict else logits