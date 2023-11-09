import argparse

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *
# from utils import load_config

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *
from GeospatialFM.loss import *

from tqdm import trange


args = get_args_parser().parse_args()
# args.debug = True
if args.debug:
    logging.basicConfig(level=logging.INFO)
cfg, _ = setup(args)

training_args = dict(
    device = args.device,
    precision = None,
    accum_freq = cfg['TRAINER']['gradient_accumulation_steps'],
    grad_clip_norm = None,
    log_every_n_steps = cfg['TRAINER']['logging_steps'],
    wandb = cfg['TRAINER']['report_to'] == 'wandb',
    batch_size = cfg['TRAINER']['per_device_train_batch_size'],
    val_frequency = 1,
    epochs = cfg['TRAINER']['num_train_epochs'],
    save_logs = True,
    checkpoint_path = cfg['TRAINER']['logging_dir'],
    mask_ratio = cfg['MODEL']['mask_ratio']
)
training_args = argparse.Namespace(**training_args)
training_args.device = f'cuda:{training_args.device}'

model = construct_mae(cfg.MODEL)
model = model.to(training_args.device)

data = get_data(cfg)

steps = data['train'].dataloader.num_batches * cfg['TRAINER']['num_train_epochs']
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAINER']['learning_rate'], weight_decay=cfg['TRAINER']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-7)
loss = [MAELoss(**cfg.LOSS['MAE']), MultiModalCELoss()]

for epoch in trange(training_args.epochs):
    train_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args)
    evaluate(model, data, loss, epoch, training_args, val_split='val')
evaluate(model, data, loss, epoch, training_args, val_split='test')
# save model
torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'model.pth'))