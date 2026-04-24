from GeospatialFM.models.DINOv3.dinov3_models import *
from transformers import PretrainedConfig, PreTrainedModel
import glob
import os
from loguru import logger

class DINOv3Config(PretrainedConfig):
    model_type = "dinov3"

    def __init__(self,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 ffn_ratio=4,
                 image_size=128,
                 in_chans=155,
                 layerscale_init=0.1,
                 n_storage_tokens=4,
                 mask_k_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.image_size = image_size
        self.in_chans = in_chans
        self.layerscale_init = layerscale_init
        self.n_storage_tokens = n_storage_tokens
        self.mask_k_bias = mask_k_bias
        
class DINOv3Encoder(DinoVisionTransformer):
    config_class = DINOv3Config
    model_type = "dinov3"
    def __init__(self, config: DINOv3Config):
        super().__init__(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            layerscale_init=config.layerscale_init,
            n_storage_tokens=config.n_storage_tokens,
            mask_k_bias=config.mask_k_bias,
        )
        
    def forward_encoder(self, x, **kwargs):
        cls_token, features = super().forward(x=x, is_training=True)['x_prenorm'][:, :1], super().forward(x=x, is_training=True)['x_prenorm'][:, 1+self.n_storage_tokens:]
        features = torch.cat([cls_token, features], dim=1)
        return features
    
    def load_pretrained_weights(self, pretrained_model_dir):
        pretrained_model_paths = glob.glob(os.path.join(pretrained_model_dir, "*.pth"))
        pretrained_model_paths.sort()
        pretrained_model_path = pretrained_model_paths[-1]
        
        state_dict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        # initialize the pretrained patch_embed.proj.weight and patch_embed.proj.bias
        state_dict['patch_embed.proj.weight'] = self.patch_embed.proj.weight
        state_dict['patch_embed.proj.bias'] = self.patch_embed.proj.bias
                
        super().load_state_dict(state_dict)
        # logger.info("Load pretrained DINOv3 Encoder successfully!")