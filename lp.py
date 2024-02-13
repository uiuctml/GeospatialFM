import argparse
from  torch.utils.data import DataLoader, Dataset

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *
from GeospatialFM.loss import *

from tqdm import trange
import random
import pandas as pd
import pickle

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
        print_log = False,
        checkpoint_path = cfg['TRAINER']['logging_dir'],
        mask_ratio = cfg['MODEL']['mask_ratio'],
    )
    training_args = argparse.Namespace(**vars(args), **training_args)
    assert training_args.distributed == False, 'Distributed training is not supported for linear probing.'

    random_seed(0, args.rank)
    models = construct_downstream_models(cfg)

    model = models[args.finetune_modal]
    feature_extractor = model.encoder.requires_grad_(False)
    task_head = model.head
    feature_extractor = feature_extractor.to(device)

    cache_dir = os.path.join(cfg['LOGGER']['dir'], 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cfg.DATASET['name']}_{cfg['NAME']}")
    os.makedirs(cache_path, exist_ok=True)
    print(f'cache path: {cache_path}')
    
    data = get_data(cfg, ddp=training_args.distributed)
    parse_splits = ['train', 'val', 'test']
    data_feat = dict()
    # check if chache_path is empty
    for split in parse_splits:
        if os.path.exists(os.path.join(cache_path, f"{split}_0.pkl")) or os.path.exists(os.path.join(cache_path, f"{split}.pkl")):
            print(f'Loading cached {split} features...')
            data_split = load_features(cache_path, split=split)
            data_feat[split] = data_split
        else:
            print(f'Extracting {split} features...')
            data_split = extract_features(feature_extractor, data[split], training_args, split=split, cache_path=cache_path)
            data_feat[split] = data_split
    # test_data = data['test'].dataloader
    del data
    del feature_extractor
    data = data_feat

    for split in parse_splits:
        dataset = DictDataset(data[split])
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=shuffle, num_workers=8, pin_memory=True)
        dataloader.num_samples = len(dataset)
        dataloader.num_batches = len(dataloader)
        data[split] = DataInfo(dataloader)
    # data['test'] = test_data
    print(f"Training Samples: {data['train'].dataloader.num_samples}\tValidation Samples: {data['val'].dataloader.num_samples}\tTest Samples: {data['test'].dataloader.num_samples}")

    model = task_head.to(device)
    steps = data['train'].dataloader.num_batches // cfg.TRAINER['gradient_accumulation_steps'] * cfg['TRAINER']['num_train_epochs']
    warmup_steps = data['train'].dataloader.num_batches // cfg.TRAINER['gradient_accumulation_steps'] * cfg['TRAINER']['warmup_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAINER']['learning_rate'], weight_decay=0)

    logging.info(f"Linear Probing Hyperparameters: LR: {cfg['TRAINER']['learning_rate']}")
    if training_args.save_logs:
        with open(os.path.join(training_args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write("===================================\n")
                f.write(f"Linear Probing Hyperparameters: LR: {cfg['TRAINER']['learning_rate']}\n")
                f.write("===================================\n")
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-7)
    scheduler = get_scheduler(cfg['TRAINER']['lr_scheduler_type'], optimizer, cfg['TRAINER']['learning_rate'], warmup_steps, steps, cfg['TRAINER']['scheduler_kwargs'])
    print(f"Task type: {cfg.DATASET['task_type']}")
    loss = get_loss(cfg.DATASET['task_type'])
    print(f"Evaluation metric: {cfg.DATASET['eval_metric']}")

    input_keyword = 'feature'
    if cfg.DATASET['task_type'] == 'change_detection':
        target_keyword = 'mask'
    else:
        target_keyword = 'label'

    for epoch in trange(training_args.epochs):
        finetune_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args, input_keyword, target_keyword)
        evaluate_finetune(model, data, loss, epoch, training_args, val_split='val', eval_metric=cfg.DATASET['eval_metric'], input_keyword=input_keyword, target_keyword=target_keyword)

    final_metrics = evaluate_finetune(model, data, loss, epoch+1, training_args, val_split='test', eval_metric=cfg.DATASET['eval_metric'], input_keyword=input_keyword, target_keyword=target_keyword)
    if training_args.save_csv and is_master(args):
        save_dict = dict(
            epochs = training_args.epochs,
            lr = cfg['TRAINER']['learning_rate'],
            weight_decay = cfg['TRAINER']['weight_decay'],
            **final_metrics
        )
        save_path = os.path.join(training_args.checkpoint_path, 'lp_metrics.csv')
        df_new = pd.DataFrame(save_dict, index=[0])
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(save_path, index=False)
    # save model
    if is_master(args):
        torch.save(model.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'lp_model.pth'))