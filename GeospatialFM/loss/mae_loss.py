import torch
import torch.nn as nn
from torch.nn import functional as F

class MAELoss(nn.Module):
    def __init__(self, recon_all=True, cross_modal_recon=False, channel_reweight=False, scale=1.0, radar_scale=1.0):
        super().__init__()
        self.recon_all = recon_all
        self.cross_modal_recon = cross_modal_recon
        assert not (recon_all and cross_modal_recon)
        self.channel_reweight = channel_reweight # TODO: implement this
        self.lambda_ = scale
        self.radar_lambda_ = radar_scale

    def _forward_mse_one_modal(self, recon, target, mask):
        loss = (recon - target).abs()
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if mask.sum() == 0:
            return loss.mean() # if no mask, mean loss on all patches
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, optical_mask, radar_mask, 
                    optical_recon, radar_recon,
                    optical_target, radar_target,
                    output_dict=False, **kwargs):
        if self.recon_all:
            if self.channel_reweight:
                optical_dim = optical_target.shape[-1]
                radar_dim = radar_target.shape[-1]
                scale = optical_dim / radar_dim
                radar_recon[:, :, -radar_dim:] *= scale
                optical_recon[:, :, -radar_dim:] *= scale
                radar_target *= scale
            combined_target = torch.cat([optical_target, radar_target], dim=-1)
            optical_target = radar_target = combined_target
        elif self.cross_modal_recon:
            optical_target, radar_target = radar_target, optical_target
        optical_mse = self._forward_mse_one_modal(optical_recon, optical_target, optical_mask) * self.lambda_
        radar_mse = self._forward_mse_one_modal(radar_recon, radar_target, radar_mask) * self.lambda_ * self.radar_lambda_
        if output_dict:
            return dict(optical_mse=optical_mse, radar_mse=radar_mse)
        return optical_mse, radar_mse
