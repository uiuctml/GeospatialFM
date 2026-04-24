from GeospatialFM.models.SpecViT.specvit import *
from transformers import PretrainedConfig, PreTrainedModel
import glob
import os
from loguru import logger

class SpecViTConfig(PretrainedConfig):
    model_type = "specvit"

    def __init__(self, 
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 token_patch_size=4,
                 num_classes=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_patch_size = token_patch_size
        self.num_classes = num_classes

class SpecViTEncoder(SpecVisionTransformer):
    config_class = SpecViTConfig
    model_type = "specvit"
    def __init__(self, config: SpecViTConfig):
        super().__init__(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            token_patch_size=config.token_patch_size,
            num_classes=config.num_classes,
        )
        
    def forward_encoder(self, x, **kwargs):
        features, cls_token = super().get_intermediate_layers(x, norm=True, return_prefix_tokens=True)[0]
        features = torch.cat([cls_token, features], dim=1)
        return features
    
    def load_pretrained_weights(self, pretrained_model_dir):
        # pretrained_model_paths = glob.glob(os.path.join(pretrained_model_dir, "pytorch_model.bin"))
        # pretrained_model_paths.sort()
        # pretrained_model_path = pretrained_model_paths[-1]
        pretrained_model_path = pretrained_model_dir
        
        state_dict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        _state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items() if k.startswith("vit.spectral_adapter") or k.startswith("vit.vit_core")}
        
        super().load_state_dict(_state_dict)
        # logger.info("Load pretrained SpectralViT Encoder successfully!")