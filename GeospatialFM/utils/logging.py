import wandb

def init_wandb(cfg):
    wandb_cfg = cfg['LOGGER']
    wandb.init(entity=wandb_cfg['entity'],
               project=wandb_cfg['project'] + cfg['DATASET']['name'],
               name = cfg['NAME'],
               reinit=True,
               dir=wandb_cfg['dir'],
               settings=wandb.Settings(_disable_stats=True, _disable_meta=True),  # disable logging of system metrics
               )
    print('W&B run name:', wandb.run.name)