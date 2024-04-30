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
        elif loss_name == 'SigLipLoss':
            loss_list.append(SigLipLoss(**loss_kwargs))
        elif loss_name == 'Spectral':
            loss_list.append(SpectralInterpolationLoss(**loss_kwargs))
        elif loss_name == 'BCE':
            loss_list.append(MultilabelBCELoss(**loss_kwargs))
        elif loss_name == 'CrossModalLoss':
            loss_list.append(CrossModalMSELoss(**loss_kwargs))
        elif loss_name == 'MSE':
            loss_list.append(MSELoss(**loss_kwargs))
        else:
            raise NotImplementedError
    return loss_list

def get_loss(task_type):
    if task_type == 'classification':
        return CrossEntropyLoss()
    elif task_type == 'change_detection':
        return MultilabelBCELoss()
    elif task_type == 'multilabel':
        return MultiLabelSoftMarginLoss()
    elif task_type == 'regression':
        return MSELoss()
    else:
        raise NotImplementedError