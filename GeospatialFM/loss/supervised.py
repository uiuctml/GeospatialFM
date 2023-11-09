import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiModalCELoss(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.lambda_ = scale

    def forward(self, optical_logits, radar_logits, labels,
                    output_dict=False, **kwargs):
        optical_ce = F.cross_entropy(optical_logits, labels) * self.lambda_
        radar_ce = F.cross_entropy(radar_logits, labels) * self.lambda_
        if output_dict:
            return dict(optical_ce=optical_ce, radar_ce=radar_ce)
        return optical_ce, radar_ce