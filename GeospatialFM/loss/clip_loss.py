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
    

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            scale=1.0,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir
        self.lambda_ = scale

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, optical_cls_token, radar_cls_token, logit_scale, logit_bias=None):
        logits = logit_scale * optical_cls_token @ radar_cls_token.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, optical_cls_token, radar_cls_token, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(optical_cls_token, radar_cls_token, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            optical_cls_token.device,
            optical_cls_token.dtype,
            optical_cls_token.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / optical_cls_token.shape[0]
        return loss

    def forward(self, optical_cls_token, radar_cls_token, logit_scale, logit_bias, output_dict=False, **kwargs):
        loss = self._loss(optical_cls_token, radar_cls_token, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                radar_cls_token_to_right = radar_cls_token_to_left = radar_cls_token
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    radar_cls_token_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        radar_cls_token_to_left,
                        radar_cls_token_to_right,
                    )

                    for f in radar_cls_token_recv:
                        loss += self._loss(
                            optical_cls_token,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    radar_cls_token_to_left, radar_cls_token_to_right = radar_cls_token_recv

                if remainder:
                    radar_cls_token_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, radar_cls_token_to_right)

                    loss += self._loss(
                        optical_cls_token,
                        radar_cls_token_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                radar_cls_token_to_right = radar_cls_token
                for i in range(self.world_size - 1):
                    radar_cls_token_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, radar_cls_token_to_right)

                    loss += self._loss(
                        optical_cls_token,
                        radar_cls_token_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    radar_cls_token_to_right = radar_cls_token_from_left

        loss *= self.lambda_
        return {"contrastive_loss": loss} if output_dict else loss