import argparse

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *
from GeospatialFM.loss import *

from tqdm import trange
import random
import pandas as pd

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    args.debug = True
    args.finetune = True
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
    args.finetune_modal = args.finetune_modal.upper()

    training_args = dict(
        # device_ids = args.device_ids,
        # device = 'cpu',
        precision = 'amp_bf16',
        accum_freq = cfg['TRAINER']['gradient_accumulation_steps'],
        grad_clip_norm = None,
        log_every_n_steps = cfg['TRAINER']['logging_steps'],
        wandb = cfg['TRAINER']['report_to'] == 'wandb',
        batch_size = cfg['TRAINER']['per_device_train_batch_size'],
        val_frequency = 1,
        epochs = cfg['TRAINER']['num_train_epochs'],
        save_logs = True,
        save_csv = True,
        print_log = True,
        checkpoint_path = cfg['TRAINER']['logging_dir'],
        mask_ratio = cfg['MODEL']['mask_ratio'],
        dataset = cfg['DATASET']['name'],
        head_type = cfg['DATASET']['task_head_kwargs']['head_type'],
    ) # CHANGE
    training_args = argparse.Namespace(**vars(args), **training_args)
    # training_args.device = f'cuda:{training_args.device_ids[0]}'

    random_seed(0, args.rank)
    models = construct_downstream_models(cfg)

    model = models[args.finetune_modal]
    if training_args.lpft:
        head_path = os.path.join(cfg['TRAINER']['output_dir'], 'lp_model.pth')
        assert os.path.exists(head_path), f'LP model not found at {head_path}'
        head_state_dict = torch.load(head_path, map_location='cpu')
        # model.head.load_state_dict(head_state_dict, strict=True)
        model.load_state_dict(head_state_dict, strict=True)

    model = model.to(training_args.device)
    # model.encoder.patch_embed.requires_grad_(False)

    random_seed(0, args.rank)
    if training_args.distributed:
        ddp_args = {'find_unused_parameters': False } # TODO: add ddp args
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # if len(training_args.device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=training_args.device_ids)

    data = get_data(cfg, ddp=training_args.distributed)
    
    steps = data['train'].dataloader.num_batches // cfg.TRAINER['gradient_accumulation_steps'] * cfg['TRAINER']['num_train_epochs']
    warmup_steps = data['train'].dataloader.num_batches // cfg.TRAINER['gradient_accumulation_steps'] * cfg['TRAINER']['warmup_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAINER']['learning_rate'], weight_decay=cfg['TRAINER']['weight_decay'])

    logging.info(f"Hyperparameters: LR: {cfg['TRAINER']['learning_rate']}, weight decay: {cfg['TRAINER']['weight_decay']}")
    if is_master(args) and training_args.save_logs:
        with open(os.path.join(training_args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write("===================================\n")
                f.write(f"Hyperparameters: LR: {cfg['TRAINER']['learning_rate']}, weight decay: {cfg['TRAINER']['weight_decay']}\n")
                f.write("===================================\n")
    scheduler = get_scheduler(cfg['TRAINER']['lr_scheduler_type'], optimizer, cfg['TRAINER']['learning_rate'], warmup_steps, steps, cfg['TRAINER']['scheduler_kwargs'])
    print(f"Task type: {cfg.DATASET['task_type']}")
    loss = get_loss(cfg.DATASET['task_type'])
    print(f"Evaluation metric: {cfg.DATASET['eval_metric']}")
    if training_args.lpft:
        evaluate_finetune(model, data, loss, 0, training_args, val_split='val', eval_metric=cfg.DATASET['eval_metric'])

    if cfg.DATASET['task_type'] == 'change_detection':
        input_keyword = 'images'
        target_keyword = 'mask'
    else:
        input_keyword = 'image' # TODO: multi-modal later
        target_keyword = 'label'

    for epoch in trange(training_args.epochs):
        finetune_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args, input_keyword, target_keyword)
        evaluate_finetune(model, data, loss, epoch, training_args, val_split='val', eval_metric=cfg.DATASET['eval_metric'], input_keyword=input_keyword, target_keyword=target_keyword)
        # if cfg.TRAINER.save_frequency > 0 and (epoch + 1) % cfg.TRAINER.save_frequency == 0:
            # torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], f'ft_ckpt_epoch{epoch+1}.pth'))
    final_metrics = evaluate_finetune(model, data, loss, epoch+1, training_args, val_split='test', eval_metric=cfg.DATASET['eval_metric'], input_keyword=input_keyword, target_keyword=target_keyword)
    if training_args.save_csv and is_master(args):
        save_dict = dict(
            epochs = training_args.epochs,
            lr = cfg['TRAINER']['learning_rate'],
            weight_decay = cfg['TRAINER']['weight_decay'],
            **final_metrics
        )
        save_path = os.path.join(training_args.checkpoint_path, 'ft_metrics.csv')
        df_new = pd.DataFrame(save_dict, index=[0])
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(save_path, index=False)
    # save model
    # if is_master(args):
    #     torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'ft_model.pth'))