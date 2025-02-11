from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .UPerNet import UPerNet
from transformers import PretrainedConfig
from .spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder
import torch.nn.functional as F
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

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
        pos_chan_embed_residual: bool = True,
        return_dict: bool = False,
        use_perception_field_mask: bool = False,
        attention_radius: int = 640,
        num_experts: int = None,
        use_moe: bool = False,
        topk: int = None,
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
        self.mask_ratio = 0
        self.channel_mask_ratio = 0
        self.pretrain = False
        self.num_experts = num_experts if num_experts is not None else 0
        self.use_moe = use_moe if self.num_experts > 0 else False
        self.topk = topk
        
        # Perception field mask
        self.use_perception_field_mask = use_perception_field_mask
        self.attention_radius = attention_radius
        
        # Positional channel embedding residual
        self.pos_chan_embed_residual = pos_chan_embed_residual

class LESSWithProjectionConfig(LESSViTEncoderConfig):
    model_type = "less_with_projection"
    
    def __init__(self, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        
class LESSWithUPerNetConfig(LESSViTEncoderConfig):
    model_type = "less_with_uper_net"
    
    def __init__(self, num_labels=2, image_size=256, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.image_size = image_size
        
class MoELinearHead(nn.Module):
    def __init__(self, embed_dim, num_labels, num_experts, topk=3):
        super().__init__()
        self.classifier = nn.ModuleList([nn.Linear(embed_dim, num_labels) for _ in range(num_experts)])
        self.gate = nn.Linear(embed_dim, num_experts)
        self.topk = topk
        self.num_experts = num_experts

    def forward(self, features, labels=None):
        if self.topk == -1:
            self.topk = features.shape[1]
        gate_score = self.gate(features) # [batch_size, n_channels, num_experts]
        gate_score = gate_score.permute(0, 2, 1) # [batch_size, num_experts, n_channels]
        gate_prob = F.softmax(gate_score, dim=-1) # [batch_size, num_experts, n_channels]
        
        expert_logits = []
        for i in range(self.num_experts):
            topk_values, topk_indices = torch.topk(gate_prob[:, i, :], self.topk, dim=-1) # [batch_size, topk]
            topk_features = features.gather(dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])) # [batch_size, topk, embed_dim]
            topk_values = F.softmax(topk_values, dim=-1).unsqueeze(-1) # [batch_size, topk, 1]
            logits = self.classifier[i](topk_features) # [batch_size, topk, num_labels]
            logits = (logits * topk_values).sum(dim=1) # [batch_size, num_labels]
            expert_logits.append(logits)
            
        # soft vote
        logits = torch.stack(expert_logits, dim=-1).mean(dim=-1) # [batch_size, num_labels]
        
        return {'logits': logits}
    
class LinearHead(nn.Module):
    def __init__(self, embed_dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_labels)
        
    def forward(self, features, labels=None):
        if len(features.shape) == 2:
            logits = self.classifier(features)
        else:
            logits = self.classifier(features[:, 0, :])
            
        return {'logits': logits}

class LESSWithTaskHead(PreTrainedModel):
    main_input_name = ['optical', 'radar']
    def __init__(self, config):
        super().__init__(config)
    
    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if isinstance(self.main_input_name, list):
            tokens = 0
            for main_input_name in self.main_input_name:
                if main_input_name in input_dict:
                    tokens += input_dict[main_input_name].numel()
            return tokens
        elif self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0
    
    def load_pretrained_encoder(self, pretrained_model_path):
        from safetensors import safe_open
        with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("encoder."):
                    # Get the corresponding key in target model
                    param = f.get_tensor(key)
                    self.state_dict()[key].copy_(param)

class LESSWithProjection(LESSWithTaskHead):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = SpatialSpectralLowRankViTEncoder(config)
        self.classifier = MoELinearHead(config.embed_dim, config.num_labels, config.num_experts, config.topk) if config.use_moe else LinearHead(config.embed_dim, config.num_labels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10, labels=None,
    ) -> Union[Tuple, dict]:
        
        # Get encoder outputsp
        outputs = self.encoder(optical, radar, optical_channel_wv, radar_channel_wv, spatial_resolution)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        else:
            outputs = outputs.last_hidden_state
            
        # Use the [CLS] token
        pooled_output = outputs[:, :, 0]
        
        # Get logits
        logits = self.classifier(pooled_output)['logits']

        return {"logits": logits} if self.config.return_dict else logits

class MoEUpperNet(nn.Module):
    def __init__(self, num_classes, image_size, embed_dim, num_experts=1, topk=3):
        super().__init__()
        self.upper_net = nn.ModuleList([UPerNet(num_classes, image_size, debug=False) for _ in range(num_experts)])
        self.gate = nn.Linear(embed_dim, num_experts)
        self.topk = topk
        self.num_experts = num_experts
        
    def forward(self, features):
        # features: [batch_size, num_channels, num_patches, embed_dim]
        if self.topk == -1:
            self.topk = features.shape[1]
            
        cls_tokens, features = features[:, :, 0, :], features[:, :, 1:, :]
        gate_score = self.gate(cls_tokens) # [batch_size, num_channels, num_experts]
        gate_score = gate_score.permute(0, 2, 1) # [batch_size, num_experts, num_channels]
        gate_prob = F.softmax(gate_score, dim=-1) # [batch_size, num_experts, num_channels]
        
        expert_logits = []
        for i in range(self.num_experts):
            # features: [batch_size, num_channels, num_patches, embed_dim]
            topk_values, topk_indices = torch.topk(gate_prob[:, i, :], self.topk, dim=-1) # [batch_size, topk]
            gather_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, features.shape[2], features.shape[3]) # [batch_size, topk, num_patches, 1]
            topk_features = torch.gather(features, dim=1, index=gather_indices) # [batch_size, topk, num_patches, embed_dim]
            topk_values = F.softmax(topk_values, dim=-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, features.shape[2], features.shape[3]) # [batch_size, topk, num_patches, 1]
            topk_features = (topk_features * topk_values).sum(dim=1) # [batch_size, num_patches, embed_dim]
            logits = self.upper_net[i](topk_features)
            expert_logits.append(logits)
        
        logits = torch.stack(expert_logits, dim=-1).mean(dim=-1)
        
        return logits
        
class LESSWithUPerNet(LESSWithTaskHead):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = SpatialSpectralLowRankViTEncoder(config)
        if config.use_moe:
            self.decoder = MoEUpperNet(config.num_labels, config.image_size, config.embed_dim, config.num_experts, config.topk)
        else:
            self.decoder = UPerNet(
                num_classes=config.num_labels,
                image_size=config.image_size,
                debug=False
            ) 
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, 
        optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10, labels=None,
    ) -> Union[Tuple, dict]:
        # Get encoder outputs
        outputs = self.encoder(optical, radar, optical_channel_wv, radar_channel_wv, spatial_resolution)
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state
        
        if isinstance(self.decoder, UPerNet):
            # Get segmentation logits
            logits = self.decoder(hidden_states[:, 0, 1:]) # [batch_size, num_patches, embed_dim]
        else:
            # Get segmentation logits
            logits = self.decoder(hidden_states) # [batch_size, num_channels, num_patches, embed_dim]

        return {"logits": logits} if self.config.return_dict else logits