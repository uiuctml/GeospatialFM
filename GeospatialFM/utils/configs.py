# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os

from omegaconf import OmegaConf

from .logging import init_wandb
from GeospatialFM.configs import default_config


def write_config(cfg, output_dir, name="config.yaml"):
    OmegaConf.to_yaml(cfg)
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    cfg['NAME'] = args.exp_name if args.exp_name is not None else cfg['MODEL']['name'].replace('/', '')
    os.makedirs(args.output_dir, exist_ok=True)
    if cfg['TRAINER']['report_to'] == 'wandb':
        init_wandb(cfg)
    if args.save_config:
        write_config(cfg, args.output_dir, name=f"{cfg['NAME']}-{cfg['DATASET']['name']}.yaml")
    return cfg