from .utils import *
from .modeling import *
import os.path as osp

def get_model(model_cfg):
    base_model = get_tgm_model(model_cfg)
    # remove the last layer "head"
    base_model.head = nn.Identity()
    task_head = ClassificationHead(**model_cfg)
    model = ViTEncoderDecoder(base_model, task_head, lp=model_cfg['lp'])
    return model