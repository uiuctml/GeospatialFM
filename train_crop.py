import argparse

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *
# from utils import load_config

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *

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
    checkpoint_path = cfg['TRAINER']['logging_dir']
)
training_args = argparse.Namespace(**training_args)
training_args.device = f'cuda:{training_args.device}'

model = CustomCROP(cfg['MODEL'], cfg['SAR_MODEL'], output_dict=True)
model = model.to(training_args.device)

data = get_data(cfg)

steps = data['train'].dataloader.num_batches * cfg['TRAINER']['num_train_epochs']
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAINER']['learning_rate'], weight_decay=cfg['TRAINER']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-7)
loss = CustomCropLoss()

for epoch in trange(training_args.epochs):
    train_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args)
    evaluate(model, data, loss, epoch, training_args)

# save model
torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'model.pth'))