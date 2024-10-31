from GeospatialFM.models.UPerNet import UPerNet
import torch.nn as nn
import torch

class SpatialSpectralLowRankViTWithTaskHead(nn.Module):
    def __init__(self, encoder, task_head):
        super().__init__()
        self.encoder = encoder
        self.task_head = task_head

    def forward(self, optical, radar, optical_channel_wv, radar_channel_wv, spatial_resolution=10):
        # Pass all inputs to the encoder
        features = self.encoder(
            optical=optical,
            radar=radar,
            optical_channel_wv=optical_channel_wv,
            radar_channel_wv=radar_channel_wv,
            spatial_resolution=spatial_resolution
        )
        # Pass encoder output through task head
        outputs = self.task_head(features)
        
        return outputs

def get_task_head(task_type, **kwargs): # TODO
    print(f"Constructing {task_type} head...")
    in_features = kwargs.pop("in_features", 768)
    num_classes = kwargs.pop("num_classes",2)
    use_bias = kwargs.pop("use_bias", True)
    kernel_size = kwargs.pop("kernel_size", 256)
    image_size = kwargs.pop("image_size", 128)
    if task_type == 'classification' or task_type == 'multilabel':
        head = nn.Linear(in_features=in_features, out_features=num_classes, bias=use_bias)
    elif task_type == 'segmentation':
        head = UPerNet(num_classes, kernel_size=kernel_size, image_size=image_size)
    else:
        raise NotImplementedError
    return head

