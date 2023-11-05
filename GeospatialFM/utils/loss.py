import torch
import torch.nn as nn
from torch.nn import functional as F

class CropLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

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

    def get_logits(self, image_features, radar_features, logit_scale):
        logits_per_image = logit_scale * image_features @ radar_features.T
        logits_per_radar = logit_scale * radar_features @ image_features.T
        
        return logits_per_image, logits_per_radar

    def forward(self, image_features, radar_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_radar = self.get_logits(image_features, radar_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_radar, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    

class CustomCropLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            loss_scale=1.0
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        # loss scale
        self.loss_scale = loss_scale

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

    def get_logits(self, image_features, radar_features, logit_scale):
        logits_per_image = logit_scale * image_features @ radar_features.T
        logits_per_radar = logit_scale * radar_features @ image_features.T
        
        return logits_per_image, logits_per_radar

    def forward(self, image_features, radar_features, logit_scale, image_logits, radar_logits, gt_labels, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_radar = self.get_logits(image_features, radar_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_radar, labels)
        ) / 2

        image_loss = F.cross_entropy(image_logits, gt_labels) * self.loss_scale
        radar_loss = F.cross_entropy(radar_logits, gt_labels) * self.loss_scale

        total_loss = contrastive_loss + image_loss + radar_loss

        return {"contrastive_loss": contrastive_loss, "image_loss": image_loss, "radar_loss": radar_loss} if output_dict else total_loss