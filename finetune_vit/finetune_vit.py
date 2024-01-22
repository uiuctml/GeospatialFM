import argparse

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *
from GeospatialFM.loss import *

from tqdm import trange
import random

def finetune_one_epoch(model, data, loss, epoch, optimizer, scheduler, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision) # GeospatialFM/utills/precision.py
    model.train()

    data['train'].set_epoch(epoch)

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    args.debug = True # in configs.py, if args.debug sets to true, then report to None
    args.finetune = True
    if args.debug:
        logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    cfg, _ = setup(args)
    args.finetune_modal = args.finetune_modal.upper() # default set to "OPTICAL"
    
    training_args = dict(
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
    )

    training_args = argparse.Namespace(**vars(args), **training_args) # training_args will overwrite any conflict args with args

    random_seed(0, args.rank)
    if training_args.distributed:
        ddp_args={}
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], ** ddp_args
        )

    """
    data is a dict w/ train(DataInfo), val(DataInfo), test(DataInfo)
    Datainfo, get_data(): GeospatialFM/utils/data.py
    DataInfo class: dataloader, sampler, shared_epoch
    """
    data = get_data(cfg, ddp=training_args.distributed)

    # calculate steps and warmup steps
    steps = data['train'].dataloader.num_batches * cfg['TRAINER']['num_train_epochs']
    warmup_steps = data['train'].dataloader.num_batches * cfg['TRAINER']['warmup_epochs']
    
    # 
    optimizer = torch.optim.Adam