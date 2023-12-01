from .clip_loss import *
from .mae_loss import *
from .supervised import *

def get_loss_list(loss_cfg):
    loss_list = []
    for loss_name, loss_kwargs in loss_cfg.items():
        if loss_name == 'MAE':
            loss_list.append(MAELoss(**loss_kwargs))
        elif loss_name == 'MMCE':
            loss_list.append(MultiModalCELoss(**loss_kwargs))
        elif loss_name == 'CLIP':
            loss_list.append(ClipLoss(**loss_kwargs))
        elif loss_name == 'CE':
            loss_list.append(CrossEntropyLoss(**loss_kwargs))
        else:
            raise NotImplementedError
    return loss_list