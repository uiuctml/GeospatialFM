# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os

from omegaconf import OmegaConf

from .logging import init_wandb
from GeospatialFM.configs import default_config

COMMON_ACRONYM = {
    "learning_rate": 'lr',
    'weight_decay': 'wd',
    'per_device_train_batch_size': 'bs',
}

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
    # setup configs
    cfg = get_cfg_from_args(args)
    # setup the experiment name
    cfg['NAME'] = cfg['MODEL']['architecture'].replace('/', '')
    if args.opts is not None:
        for new_attr in args.opts:
            name, val = new_attr.split('=')
            name = name.split('.')[-1]
            if name == 'freeze_encoder':
                continue
            name = COMMON_ACRONYM.get(name, name)
            args.exp_name = args.exp_name + f'_{name}{val}' if args.exp_name is not None else f'{name}{val}'
    if args.exp_name is not None:
        cfg['NAME'] += f"_{args.exp_name}"
    if cfg['MODEL']['freeze_encoder']:
        cfg['NAME'] += f"_lp"
    # setup output directory
    cfg['TRAINER']['output_dir'] += f'/{cfg["NAME"]}'
    cfg['TRAINER']['logging_dir'] += f'/{cfg["NAME"]}'
    # setup logger
    if args.debug:
        cfg['TRAINER']['report_to'] = None
    run = init_wandb(cfg) if cfg['TRAINER']['report_to'] == 'wandb' else None
    return cfg, run