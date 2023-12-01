import torch
import torch.nn as nn
from torch.nn import functional as F

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            scale=1.0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.lambda_ = scale

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, optical_cls_token, radar_cls_token, logit_scale):
        logits_per_optical = logit_scale * optical_cls_token @ radar_cls_token.T
        logits_per_radar = logit_scale * radar_cls_token @ optical_cls_token.T
        
        return logits_per_optical, logits_per_radar

    def forward(self, optical_cls_token, radar_cls_token, logit_scale, output_dict=False, **kwargs):
        device = optical_cls_token.device
        logits_per_optical, logits_per_radar = self.get_logits(optical_cls_token, radar_cls_token, logit_scale)

        labels = self.get_ground_truth(device, logits_per_optical.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_optical, labels) +
            F.cross_entropy(logits_per_radar, labels)
        ) / 2
        total_loss *= self.lambda_

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    

# TODO: SigCLIP