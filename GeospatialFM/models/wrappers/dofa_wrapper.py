from GeospatialFM.models.DOFA.dofa_model import *
from transformers import PretrainedConfig, PreTrainedModel
import glob
import os
from loguru import logger

class DOFAConfig(PretrainedConfig):
    model_type = "dofa"

    def __init__(self, img_size=128, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_classes=768,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.global_pool = global_pool
        self.num_classes = num_classes
        
class DOFAEncoder(OFAViT):
    config_class = DOFAConfig
    model_type = "dofa"
    def __init__(self, config: DOFAConfig):
        super().__init__(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            norm_layer=config.norm_layer,
            global_pool=config.global_pool,
            num_classes=config.num_classes,
        )
        
    def forward_encoder(self, x, wave_list):
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        return x
    
    def resize_pos_embed(self, pos_embed, target_pos_embed):
        class_token = pos_embed[:, :1, :]
        original_pos_embed = pos_embed[:, 1:, :]
        orig_size = int((original_pos_embed.shape[1]) ** 0.5)
        assert orig_size * orig_size == original_pos_embed.shape[1], "Original position embedding size is not a square"
        new_size = int((target_pos_embed.shape[1] - 1) ** 0.5)
        assert new_size * new_size == target_pos_embed.shape[1] - 1, "New position embedding size is not a square"
        original_pos_embed = original_pos_embed.reshape(-1, orig_size, orig_size, pos_embed.shape[-1]).permute(0, 3, 1, 2)
        # bilinear interpolate
        new_pos_embed = torch.nn.functional.interpolate(
            original_pos_embed, size=(new_size, new_size), mode='bilinear', align_corners=False)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).reshape(-1, new_size * new_size, pos_embed.shape[-1])
        new_pos_embed = torch.cat([class_token, new_pos_embed], dim=1)
        assert new_pos_embed.shape == target_pos_embed.shape, "New position embedding shape does not match target position embedding shape"
        return new_pos_embed
       
    def load_pretrained_weights(self, pretrained_model_dir):
        # pretrained_model_paths = glob.glob(os.path.join(pretrained_model_dir, "*.pth"))
        # pretrained_model_paths.sort()
        # pretrained_model_path = pretrained_model_paths[-1]
        pretrained_model_path = pretrained_model_dir
        
        state_dict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
                
        state_dict.pop('mask_token')
        original_pos_embed = state_dict['pos_embed']
        state_dict['pos_embed'] = self.resize_pos_embed(original_pos_embed, self.pos_embed.data)
        
        super().load_state_dict(state_dict)
        # logger.info("Load pretrained DOFA Encoder successfully!")