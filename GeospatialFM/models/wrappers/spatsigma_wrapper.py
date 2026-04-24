from GeospatialFM.models.SpatSigma.spatsigma_seg import SSFusionFramework as SpatSigmaSeg
from GeospatialFM.models.SpatSigma.spatsigma_cls import SSFusionFramework as SpatSigmaCls
from transformers import PretrainedConfig, PreTrainedModel
import torch
import glob
import os
from loguru import logger

class SpatSigmaMixin:
    def get_state_dict(self, spat_pretrained_path, spec_pretrained_path):
        spat_net = torch.load(spat_pretrained_path, map_location='cpu', weights_only=False)
        per_net = torch.load(spec_pretrained_path, map_location='cpu', weights_only=False)
        for k in list(spat_net['model'].keys()):
            if 'patch_embed.proj' in k:
                del spat_net['model'][k]
        for k in list(spat_net['model'].keys()):
            if 'spat_map' in k:
                del spat_net['model'][k]
        for k in list(spat_net['model'].keys()):
            if 'spat_output_maps' in k:
                del spat_net['model'][k]
        for k in list(spat_net['model'].keys()):
            if 'pos_embed' in k:
                del spat_net['model'][k]
        spat_weights = {}
        prefix = 'spat_encoder.'
        for key, value in spat_net['model'].items():
            new_key = prefix + key
            spat_weights[new_key] = value
        # per_net = torch.load((r"spec-base.pth"), map_location=torch.device('cpu'))
        model_params = self.state_dict()
        for k in list(per_net['model'].keys()):
            if 'patch_embed.proj' in k:
                del per_net['model'][k]
            if 'spat_map' in k:
                del per_net['model'][k]
            if 'fpn1.0.weight' in k:
                del per_net['model'][k]
        spec_weights = {}
        prefix = 'spec_encoder.'
        for key, value in per_net['model'].items():
            new_key = prefix + key
            spec_weights[new_key] = value
        model_params = self.state_dict()
        for k in list(spec_weights.keys()):
            if 'spec_encoder.patch_embed' in k:
                del spec_weights[k]
        merged_params = {**spat_weights, **spec_weights}
        same_parsms = {k: v for k, v in merged_params.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        return model_params
    
    def _load_pretrained_weights(self, pretrained_model_dir):
        spat_pretrained_path = glob.glob(os.path.join(pretrained_model_dir, "spat*.pth"))
        spec_pretrained_path = glob.glob(os.path.join(pretrained_model_dir, "spec*.pth"))
        spat_pretrained_path.sort()
        spec_pretrained_path.sort()
        spat_pretrained_path = spat_pretrained_path[-1]
        spec_pretrained_path = spec_pretrained_path[-1]
        state_dict = self.get_state_dict(spat_pretrained_path, spec_pretrained_path)
        return state_dict
        

class SpatSigmaConfig(PretrainedConfig):
    model_type = "spatsigma"

    def __init__(self, img_size=128,
            in_channels=155,
            patch_size=8,
            classes=128,
            model_size='base',
            **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.classes = classes
        self.model_size = model_size

class SpatSigmaSegEncoder(SpatSigmaSeg, SpatSigmaMixin):
    config_class = SpatSigmaConfig
    model_type = "spatsigma"
    def __init__(self, config: SpatSigmaConfig):
        super().__init__(
            img_size=config.img_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            classes=config.classes,
            model_size=config.model_size
        )
        
    def forward_encoder(self, x, **kwargs):
        return super().forward(x)
    
    def load_pretrained_weights(self, pretrained_model_dir):
        state_dict = self._load_pretrained_weights(pretrained_model_dir)
        super().load_state_dict(state_dict)
        # logger.info("Load pretrained SpatSigma Encoder successfully!")
    
class SpatSigmaClsEncoder(SpatSigmaCls, SpatSigmaMixin):
    config_class = SpatSigmaConfig
    model_type = "spatsigma"
    def __init__(self, config: SpatSigmaConfig):
        super().__init__(
            img_size=config.img_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            classes=config.classes,
            model_size=config.model_size
        )
        
    def forward_encoder(self, x, **kwargs):
        return super().forward(x)
    
    def load_pretrained_weights(self, pretrained_model_dir):
        state_dict = self._load_pretrained_weights(pretrained_model_dir)
        super().load_state_dict(state_dict)
        # logger.info("Load pretrained SpatSigma Encoder successfully!")
