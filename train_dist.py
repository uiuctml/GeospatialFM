import argparse

from GeospatialFM.models import *

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *
from GeospatialFM.loss import *

from tqdm import trange
import random

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

args = get_args_parser().parse_args()
# args.debug = True
if args.debug:
    logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# fully initialize distributed device environment
device = init_distributed_device(args)

cfg, _ = setup(args)

training_args = dict(
    # device_ids = args.device_ids,
    # device = device,
    precision = 'amp_bf16',
    accum_freq = cfg['TRAINER']['gradient_accumulation_steps'],
    grad_clip_norm = None,
    log_every_n_steps = cfg['TRAINER']['logging_steps'],
    wandb = cfg['TRAINER']['report_to'] == 'wandb',
    batch_size = cfg['TRAINER']['per_device_train_batch_size'],
    val_frequency = 1,
    epochs = cfg['TRAINER']['num_train_epochs'],
    save_logs = True,
    checkpoint_path = cfg['TRAINER']['logging_dir'],
    mask_ratio = cfg['MODEL']['mask_ratio'],
)

# update args with training_args
training_args = argparse.Namespace(**vars(args), **training_args)

random_seed(0, args.rank)
model = construct_mae(cfg.MODEL) # TODO: siglip has different logit scale
model = model.to(training_args.device)
# if len(training_args.device_ids) > 1:
    # model = torch.nn.DataParallel(model, device_ids=training_args.device_ids)
random_seed(0, args.rank)
if training_args.distributed:
    ddp_args = {} # TODO: add ddp args
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

data = get_data(cfg, ddp=training_args.distributed)

steps = data['train'].dataloader.num_batches * cfg['TRAINER']['num_train_epochs']
warmup_steps = data['train'].dataloader.num_batches * cfg['TRAINER']['warmup_epochs']
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAINER']['learning_rate'], weight_decay=cfg['TRAINER']['weight_decay']) # TODO: add beta
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-7) # TODO: change to warmup
scheduler = get_scheduler(cfg['TRAINER']['lr_scheduler_type'], optimizer, cfg['TRAINER']['learning_rate'], warmup_steps, steps, cfg['TRAINER']['scheduler_kwargs'])
loss = get_loss_list(cfg.LOSS)

for epoch in trange(training_args.epochs):
    train_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args)
    evaluate(model, data, loss, epoch, training_args, val_split='val')
    if cfg.TRAINER.save_frequency > 0 and (epoch + 1) % cfg.TRAINER.save_frequency == 0:
        torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], f'ckpt_epoch{epoch+1}.pth'))
evaluate(model, data, loss, epoch, training_args, val_split='test')
# save model
if is_master(args):
    torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'final_model.pth'))