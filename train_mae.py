import argparse

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *

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
    device_ids = args.device_ids,
    device = 'cpu',
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
training_args.device = f'cuda:{training_args.device_ids[0]}'

model = construct_mae(cfg.MODEL)
model = model.to(training_args.device)
if len(training_args.device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=training_args.device_ids)

data = get_data(cfg)

steps = data['train'].dataloader.num_batches * cfg['TRAINER']['num_train_epochs']
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAINER']['learning_rate'], weight_decay=cfg['TRAINER']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-7)
loss = get_loss_list(cfg.LOSS)

for epoch in trange(training_args.epochs):
    train_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args)
    evaluate(model, data, loss, epoch, training_args, val_split='val')
    if cfg.TRAINER.save_frequency > 0 and (epoch + 1) % cfg.TRAINER.save_frequency == 0:
        torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], f'ckpt_epoch{epoch+1}.pth'))
evaluate(model, data, loss, epoch, training_args, val_split='test')
# save model
torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'final_model.pth'))