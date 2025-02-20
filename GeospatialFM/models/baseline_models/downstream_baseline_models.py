from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from ..UPerNet import UPerNet
from transformers import PretrainedConfig
import torch.nn.functional as F
from typing import Dict, Any
import logging
import math

from .croma import PretrainedCROMA as CROMA
from .satmae import vit_base_patch16 as SatMAE
from .spectralgpt import vit_base_patch8_128 as SpectralGPT
from .scalemae import vit_large_patch16 as ScaleMAE
from .satmae_plus import vit_large_patch16 as SatMAE_plus
from .channel_vit import hcs_channelvit_base as ChannelViT

from .croma_landsat import PretrainedCROMA_landsat as CROMA_landsat

logger = logging.getLogger(__name__)

CKPT_PATH = "baseline_model_ckpt"

BASELINE_MODELS = {  # Lazy initialization TODO: add your baseline model here
    "croma_optical": lambda: CROMA(pretrained_path=f'{CKPT_PATH}/CROMA_base.pt', size='base', modality='optical', image_resolution=120),
    "croma_radar": lambda: CROMA(pretrained_path=f'{CKPT_PATH}/CROMA_base.pt', size='base', modality='SAR', image_resolution=120),
    "croma_multi": lambda: CROMA(pretrained_path=f'{CKPT_PATH}/CROMA_base.pt', size='base', modality='both', image_resolution=120),
    "croma_landsat_optical": lambda: CROMA_landsat(pretrained_path=f'{CKPT_PATH}/CROMA_base.pt', size='base', modality='optical', image_resolution=120),
    "satmae": lambda: SatMAE(img_size=96, patch_size=8, in_chans=10),
    "satmae_landsat": lambda: SatMAE(img_size=96, patch_size=8, in_chans=20, channel_groups=((0, 1, 2, 6, 10, 11, 12, 16), (3, 4, 5, 7, 13, 14, 15, 17), (8, 9, 18, 19))),
    "spectralgpt": lambda: SpectralGPT(),
    "scalemae": lambda: ScaleMAE(img_size=224, global_pool=True),
    "satmae++": lambda: SatMAE_plus(img_size=96, patch_size=8, in_chans=10),
    "channelvit": lambda: ChannelViT(in_chans=12),
    "channelvit_landsat": lambda: ChannelViT(in_chans=20),
}

class BaselineEncoderConfig(PretrainedConfig):
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
        model_name: str = None,
        dataset_name: str = None,
        modal: str = "optical",
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
        self.model_name = model_name
        self.dataset_name = dataset_name # for landsat
        self.modal = modal
        
        # Perception field mask
        self.use_perception_field_mask = use_perception_field_mask
        self.attention_radius = attention_radius
        
        # Positional channel embedding residual
        self.pos_chan_embed_residual = pos_chan_embed_residual

class BaselineWithProjectionConfig(BaselineEncoderConfig):
    model_type = "baseline_with_projection"
    
    def __init__(self, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        
class BaselineWithUPerNetConfig(BaselineEncoderConfig):
    model_type = "baseline_with_uper_net"
    
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
    
class BaselineWithTaskHead(PreTrainedModel):
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
        # TODO: add your model's ckpt loading code here
        if self.config.model_name in ["croma_optical", "croma_radar", "croma_multi", "croma_landsat_optical"]:
            return
        elif self.config.model_name == "satmae":
            state_dict = torch.load(f"{CKPT_PATH}/pretrain-vit-base-e199.pth")['model']
        elif self.config.model_name == "satmae_landsat":
            state_dict = torch.load(f"{CKPT_PATH}/pretrain-vit-base-e199.pth")['model']
            state_dict = load_checkpoint_with_resize(self.encoder.state_dict(), state_dict)
        elif self.config.model_name in ["spectralgpt", "spectralgpt_landsat"]:
            state_dict = torch.load(f"{CKPT_PATH}/SpectralGPT+.pth")['model']
        elif self.config.model_name in ["satmae++", "satmae++_landsat"]:
            state_dict = torch.load(f"{CKPT_PATH}/checkpoint_ViT-L_pretrain_fmow_sentinel.pth")['model']
        elif self.config.model_name in ["scalemae", "scalemae_landsat"]:
            state_dict = torch.load(f"{CKPT_PATH}/scalemae-vitlarge-800.pth")['model']
        elif self.config.model_name in ["channelvit", "channelvit_landsat"]:
            pretrained_model = torch.hub.load('insitro/ChannelViT', 'so2sat_channelvit_small_p8_with_hcs_random_split_supervised', pretrained=True)
            state_dict = pretrained_model.state_dict()
            model_state_dict = self.encoder.state_dict()
            for name, param in state_dict.items():
                if name in model_state_dict:
                    if param.shape == model_state_dict[name].shape:
                        print(f"Copying parameter: {name}")
                        model_state_dict[name].copy_(param)
                    else:
                        print(f"Resizing parameter: {name} from {param.shape} to {model_state_dict[name].shape}")
                        if name == "cls_token":
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                        elif name == "pos_embed":
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False) # 1, 384, 768
                            param_interpolated = F.interpolate(param_interpolated.permute(0, 2, 1), size=model_state_dict[name].shape[-2], mode='linear', align_corners=False) # 1, 197, 768
                            param_interpolated = param_interpolated.permute(0, 2, 1)
                        elif name == "patch_embed.proj.weight":
                            param_interpolated = F.interpolate(param, size=(model_state_dict[name].shape[-3], model_state_dict[name].shape[-2], model_state_dict[name].shape[-1]), mode='nearest')
                            param_interpolated = param_interpolated.repeat_interleave(2, dim=0)
                        elif name == "patch_embed.proj.bias":
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif name == "patch_embed.channel_embed.weight":
                            param = param.unsqueeze(0)
                            param_interpolated = F.interpolate(param, size=(model_state_dict[name].shape[-1], ), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1, 2)
                            param_interpolated = F.interpolate(param_interpolated, size=(model_state_dict[name].shape[-2],), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1,2).squeeze(0)
                        elif "norm1.weight" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "norm1.bias" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "qkv.weight" in name:
                            param = param.unsqueeze(0)
                            param_interpolated = F.interpolate(param, size=(model_state_dict[name].shape[-1], ), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1, 2)
                            param_interpolated = F.interpolate(param_interpolated, size=(model_state_dict[name].shape[-2],), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1,2).squeeze(0)
                        elif "qkv.bias" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "attn.proj.weight" in name:
                            param = param.unsqueeze(0)
                            param_interpolated = F.interpolate(param, size=(model_state_dict[name].shape[-1], ), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1, 2)
                            param_interpolated = F.interpolate(param_interpolated, size=(model_state_dict[name].shape[-2],), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1,2).squeeze(0)
                        elif "attn.proj.bias" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "norm2.weight" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "norm2.bias" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "mlp.fc1.weight" in name:
                            param = param.unsqueeze(0)
                            param_interpolated = F.interpolate(param, size=(model_state_dict[name].shape[-1], ), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1, 2)
                            param_interpolated = F.interpolate(param_interpolated, size=(model_state_dict[name].shape[-2],), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1,2).squeeze(0)
                        elif "mlp.fc1.bias" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif "mlp.fc2.weight" in name:
                            param = param.unsqueeze(0)
                            param_interpolated = F.interpolate(param, size=(model_state_dict[name].shape[-1], ), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1, 2)
                            param_interpolated = F.interpolate(param_interpolated, size=(model_state_dict[name].shape[-2],), mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.transpose(1,2).squeeze(0)
                        elif "mlp.fc2.bias" in name:
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif name == "norm.weight":
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        elif name == "norm.bias":
                            param = param.view(1, 1, -1)
                            param_interpolated = F.interpolate(param, size=model_state_dict[name].shape[-1], mode='linear', align_corners=False)
                            param_interpolated = param_interpolated.squeeze(0).squeeze(0)
                        else:
                            raise NotImplementedError
                        
                        model_state_dict[name].copy_(param_interpolated)

            return
        else: 
            raise NotImplementedError
        
        self.encoder.load_state_dict(state_dict, strict=False)

class BaselineWithProjection(BaselineWithTaskHead):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        if config.model_name == "croma":
            config.model_name = config.model_name + "_" + config.modal
        self.encoder = BASELINE_MODELS[config.model_name]()
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"Total parameters: {total_params}")
    
        self.classifier = MoELinearHead(config.embed_dim, config.num_labels, config.num_experts, config.topk) if config.use_moe else LinearHead(config.embed_dim, config.num_labels)
        
        # Initialize weights
        # self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10, labels=None,
    ) -> Union[Tuple, dict]:
        
        # TODO: add your model's forward pass code here
        if self.config.model_name == "croma_optical":
            outputs = self.encoder(optical_images=optical)['optical_GAP']
        elif self.config.model_name == "croma_radar":
            outputs = self.encoder(SAR_images=radar)['SAR_GAP']
        elif self.config.model_name == "croma_multi":
            outputs = self.encoder(SAR_images=radar, optical_images=optical)['joint_GAP']
        elif self.config.model_name in ["satmae", "spectralgpt", "satmae++", "scalemae"]:
            outputs = self.encoder(optical)['outcome']
        elif self.config.model_name in ['channelvit']:
            B = optical.shape[0]
            channels = torch.tensor([0,1,2,3,4,5,6,7,8,9,11,12], device=self.encoder.device)
            channels = channels.unsqueeze(dim=0).repeat(B, 1)
                                    
            outputs, _ = self.encoder(optical)
        else:
            raise NotImplementedError
        
        logits = self.classifier(outputs)['logits']

        return {"logits": logits} if self.config.return_dict else logits

class BaselineWithUPerNet(BaselineWithTaskHead):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        if config.model_name in ["croma", "croma_landsat"]:
            config.model_name = config.model_name + "_" + config.modal
        self.encoder = BASELINE_MODELS[config.model_name]()
        self.decoder = UPerNet(
            num_classes=config.num_labels,
            image_size=config.image_size if config.dataset_name != "landsat" else 128,
            debug=False,
            embed_dim=config.embed_dim
        )
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"Total parameters: {total_params}")
        
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # Initialize weights
        # self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, 
        optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10, labels=None,
    ) -> Union[Tuple, dict]:
        # TODO: add your model forward pass code here
        if self.config.model_name in ["croma_optical", "croma_landsat_optical"]:
            outputs = self.encoder(optical_images=optical)['optical_encodings']
        elif self.config.model_name == "croma_radar":
            outputs = self.encoder(SAR_images=radar)['SAR_encodings']
        elif self.config.model_name == "croma_multi":
            outputs = self.encoder(SAR_images=radar, optical_images=optical)['joint_encodings']
        elif self.config.model_name in ["satmae", "spectralgpt", "satmae++", "scalemae", "satmae_landsat", "spectralgpt_landsat", "satmae++_landsat", "scalemae_landsat"]:
            outputs = self.encoder(optical)['patch_embeddings']
        elif self.config.model_name in ["channelvit", "channelvit_landsat"]:
            B = optical.shape[0]
            channels = torch.tensor([0,1,2,3,4,5,6,7,8,9,11,12], device=self.encoder.device)
            channels = channels.unsqueeze(dim=0).repeat(B, 1)
                                    
            _, outputs = self.encoder(optical)
            num_channels = outputs.shape[1]
            new_num_channels = (int(math.sqrt(num_channels)))**2
            outputs = outputs[:, :new_num_channels]

        else:
            raise NotImplementedError
            
        logits = self.decoder(outputs)

        return {"logits": logits} if self.config.return_dict else logits

def load_checkpoint_with_resize(model_dict, state_dict):
    new_state_dict = {}

    for key, pretrained_weight in state_dict.items():
        if key in model_dict:
            model_weight = model_dict[key]
            if pretrained_weight.shape != model_weight.shape:
                print(f"Resizing {key}: {pretrained_weight.shape} -> {model_weight.shape}")
                new_state_dict[key] = model_weight
            else:
                new_state_dict[key] = pretrained_weight

    return new_state_dict