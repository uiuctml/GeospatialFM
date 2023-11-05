import torch
import torch.nn as nn
from typing import Tuple, Union, Optional
import numpy as np
from .utils import consturct_encoder, construct_head
import torch.nn.functional as F


class CROP(nn.Module):
    def __init__(self,
                 optical_config: dict,
                 radar_config: dict,
                 # MISC
                 init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None,
                 cast_dtype: Optional[torch.dtype] = None,
                 output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = consturct_encoder(optical_config)
        self.radar = consturct_encoder(radar_config)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    @property
    def dtype(self):
        # return the first parameter in self.visual
        return next(self.visual.parameters()).dtype

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image.type(self.dtype))
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_radar(self, radar, normalize: bool = False):
        features = self.radar(radar.type(self.dtype))
        return F.normalize(features, dim=-1) if normalize else features
    
    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            radar: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        radar_features = self.encode_radar(radar, normalize=True) if radar is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "radar_features": radar_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, radar_features, self.logit_scale.exp(), self.logit_bias
        return image_features, radar_features, self.logit_scale.exp()

        
class CustomCROP(nn.Module):
    def __init__(self,
                 optical_config: dict,
                 radar_config: dict,
                 # MISC
                 init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None,
                #  cast_dtype: Optional[torch.dtype] = None,
                 output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = consturct_encoder(optical_config)
        self.radar = consturct_encoder(radar_config)
        self.visual_head = construct_head(optical_config['head_kwargs'])
        self.radar_head = construct_head(radar_config['head_kwargs'])

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    @property
    def dtype(self):
        # return the first parameter in self.visual
        return next(self.visual.parameters()).dtype

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image.type(self.dtype))
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_radar(self, radar, normalize: bool = False):
        features = self.radar(radar.type(self.dtype))
        return F.normalize(features, dim=-1) if normalize else features
    
    def predict_image(self, image, return_logits: bool = False):
        image_features = self.encode_image(image, normalize=True)
        logits = self.visual_head(image_features)
        return torch.argmax(logits, dim=-1) if not return_logits else logits
    
    def predict_radar(self, radar, return_logits: bool = False):
        radar_features = self.encode_radar(radar, normalize=True)
        logits = self.radar_head(radar_features)
        return torch.argmax(logits, dim=-1) if not return_logits else logits
    
    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            radar: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        radar_features = self.encode_radar(radar, normalize=True) if radar is not None else None

        image_logits = self.visual_head(image_features) if image_features is not None else None
        radar_logits = self.radar_head(radar_features) if radar_features is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "radar_features": radar_features,
                "image_logits": image_logits,
                "radar_logits": radar_logits,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, radar_features, self.logit_scale.exp(), self.logit_bias, image_logits, radar_logits
        return image_features, radar_features, self.logit_scale.exp(), image_logits, radar_logits


# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int
#                  ):
#         super().__init__()

#         self.context_length = context_length

#         if isinstance(vision_layers, (tuple, list)):
#             vision_heads = vision_width * 32 // 64
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width
#             )
#         else:
#             vision_heads = vision_width // 64
#             self.visual = VisionTransformer(
#                 input_resolution=image_resolution,
#                 patch_size=vision_patch_size,
#                 width=vision_width,
#                 layers=vision_layers,
#                 heads=vision_heads,
#                 output_dim=embed_dim
#             )

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)

#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.initialize_parameters()

#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)

#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)

#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         if self.text_projection is not None:
#             nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     @property
#     def dtype(self):
#         return self.visual.conv1.weight.dtype

#     def encode_image(self, image):
#         return self.visual(image.type(self.dtype))

#     def encode_text(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

#         return x

#     def forward(self, image, text):
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         # normalized features
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logits_per_image.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text


# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)


# def build_model(state_dict: dict):
#     vit = "visual.proj" in state_dict

#     if vit:
#         vision_width = state_dict["visual.conv1.weight"].shape[0]
#         vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
#         vision_layers = tuple(counts)
#         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
#         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         vision_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32

#     embed_dim = state_dict["text_projection"].shape[1]
#     context_length = state_dict["positional_embedding"].shape[0]
#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     transformer_width = state_dict["ln_final.weight"].shape[0]
#     transformer_heads = transformer_width // 64
#     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

#     model = CLIP(
#         embed_dim,
#         image_resolution, vision_layers, vision_width, vision_patch_size,
#         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
#     )

#     for key in ["input_resolution", "context_length", "vocab_size"]:
#         if key in state_dict:
#             del state_dict[key]

#     convert_weights(model)
#     model.load_state_dict(state_dict)
#     return model.eval()