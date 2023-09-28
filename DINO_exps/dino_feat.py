# import sys
# sys.path.append('../')

import torch

import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import cv2

import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE
# from umap import UMAP
import pandas as pd
import seaborn as sns

from datasets import PASTIS_Dataset, pad_collate

PATH_TO_PASTISR = '/data/common/STIS/PASTIS-R'
dt = PASTIS_Dataset(PATH_TO_PASTISR, norm=True, target='instance', sats=['S2','S1A','S1D'], resize=448)
dl = torch.utils.data.DataLoader(dt, batch_size=16, collate_fn=pad_collate, shuffle=False)

dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14.eval()

log_dir = f'/data/common/STIS/PASTIS-R/logs/vitb14'
os.makedirs(log_dir, exist_ok=True)

# for all training samples, collect patch level features
device = 2
n_time_steps = 16
device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
dinov2_vitb14.to(device)
patch_level_feats = []
cls_feats = []
seg_ann = []
for batch_idx, ((x, dates), y) in enumerate(tqdm(dl)):
    B, N, C, H, W = x['S2'].shape
    xs = x['S2'].squeeze(1)[:, :n_time_steps, :3, :, :].reshape(B*n_time_steps, 3, H, W).to(device)
    (
        target_heatmap,
        instance_ids,
        pixel_to_instance_mapping,
        instance_bbox_size,
        object_semantic_annotation,
        pixel_semantic_annotation,
    ) = y.split((1, 1, 1, 2, 1, 1), dim=-1)

    with torch.no_grad():
        ret = dinov2_vitb14(xs, is_training=True)
    patch_level_feat = ret['x_norm_patchtokens']
    cls_feat = ret['x_norm_clstoken']
    patch_level_feat = patch_level_feat.reshape(B, n_time_steps, *patch_level_feat.shape[-2:])
    cls_feat = cls_feat.reshape(B, n_time_steps, -1)
    patch_level_feats.append(patch_level_feat.cpu())
    cls_feats.append(cls_feat.cpu())
    seg_ann.append(pixel_semantic_annotation.cpu())

patch_level_feats = torch.cat(patch_level_feats, dim=0)
seg_anns = torch.cat(seg_ann, dim=0)
cls_feats = torch.cat(cls_feats, dim=0)

# save all
torch.save(patch_level_feats, os.path.join(log_dir, f'patch_feat_c.pt'))
torch.save(seg_anns, os.path.join(log_dir, f'anns_c.pt'))
torch.save(cls_feats, os.path.join(log_dir, f'cls_feat_c.pt'))