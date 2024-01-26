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

def finetune_one_epoch(model, data, loss, epoch, optimizer, scheduler, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    model.train()

    dataloader = data['train']
    num_batches_per_epoch = len(dataloader) // args.accum_freq
    sample_digits = math.ceil(math.log(len(dataloader.dataset) + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        scheduler(step)

        # images, radar, label = batch['image'], batch['radar'], batch['label']
        feature = batch['feature']
        label = batch['label']
        feature = feature.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)  

        data_time_m.update(time.time() - end)
        
        # if args.accum_freq == 1:
        with autocast():
            model_out = model(feature)
            losses = loss(model_out, label, output_dict=True)

            total_loss = sum(losses.values())
            losses["loss"] = total_loss

        total_loss.backward()

        if (i + 1) % args.accum_freq == 0:
            optimizer.step()
            optimizer.zero_grad() 
        # else:
        #     raise NotImplementedError

        # optimizer.step()
        # scheduler.step()

        # reset gradient accum, if enabled
        # if args.accum_freq > 1:
            # accum_images, accum_radar, accum_features = [], [], {}

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(feature)
            num_samples = batch_count * batch_size * args.accum_freq
            samples_per_epoch = len(dataloader.dataset)
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            # logging.info(
            #     f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
            #     f"Data (t): {data_time_m.avg:.3f} "
            #     f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
            #     f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
            # )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
            }            

            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def evaluate_finetune(model, data, loss, epoch, args, val_split='val', eval_metric='accuracy', input_key='feature'):
    assert eval_metric in ['accuracy', 'mAP']
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    # input_dtype = get_input_dtype(args.precision)

    if val_split in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        if eval_metric == 'mAP':
            all_preds = []
            all_labels = []
        dataloader = data[val_split]
        num_samples = 0
        samples_per_val = len(dataloader.dataset)

        cumulative_losses = defaultdict(float)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input = batch[input_key]
                label = batch['label']
                input = input.to(device=device, non_blocking=True)
                label = label.to(device=device, non_blocking=True) 

                batch_size = len(input)

                with autocast():
                    model_out = model(input)
                    losses = loss(model_out, label, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                    
                    if eval_metric == 'accuracy':
                        preds = torch.argmax(model_out, dim=1)
                        image_acc = (preds == label).sum().item() / len(label)
                        losses['image_acc'] = image_acc
                    elif eval_metric == 'mAP':
                        model_out = F.sigmoid(model_out)
                        all_preds.append(model_out.cpu())
                        all_labels.append(label.cpu())
                        # _all_preds = torch.cat(all_preds, dim=0).float()
                        # _all_labels = torch.cat(all_labels, dim=0).float()
                        # mAP = average_precision_score(_all_labels.cpu().numpy(), _all_preds.cpu().numpy(), average='macro')
                        # losses['mAP'] = mAP

                num_samples += batch_size

                for key, val in losses.items():
                    if key == 'mAP': 
                        pass
                        # cumulative_losses['mAP'] = losses['mAP'] * num_samples
                    else:
                        try:
                            cumulative_losses[key] += val.item() * batch_size
                        except:
                            cumulative_losses[key] += val * batch_size

                if is_master(args) and (i % 100) == 0:
                    loss_log = f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]" 
                    loss_log += " ".join(
                        [
                            f"{key.replace('_', ' ').title()}: {val / num_samples:.6f}" 
                            for key, val in cumulative_losses.items()
                        ]
                    )
                    logging.info(loss_log)

            for key, val in cumulative_losses.items():
                metrics[key] = val / num_samples
            
            if eval_metric == 'mAP':
                _all_preds = torch.cat(all_preds, dim=0).float()
                _all_labels = torch.cat(all_labels, dim=0).float()
                mAP = average_precision_score(_all_labels.numpy(), _all_preds.numpy(), average='macro')
                metrics['mAP'] = mAP

            metrics.update({"epoch": epoch, "num_samples": num_samples})
            # if gen_loss is not None:
            #     gen_loss = cumulative_gen_loss / num_samples
            #     metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        # if 'train' in data:
        #     dataloader = data['train'].dataloader
        #     num_batches_per_epoch = dataloader.num_batches // args.accum_freq
        #     step = num_batches_per_epoch * epoch
        # else:
        #     step = None
        step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics

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
    parse_splits = ['train', 'val']
    data_feat = dict()
    # check if chache_path is empty
    if len(os.listdir(cache_path)) > 0:
        print('Loading cached features...')
        for split in parse_splits:
            cache_file = os.path.join(cache_path, f"{split}.pkl")
            with open(cache_file, 'rb') as f:
                data_split = pickle.load(f)
            data_feat[split] = data_split
    else:
        print('Extracting features...')
        for split in parse_splits:
            data_split = extract_features(feature_extractor, data[split], training_args)
            cache_file = os.path.join(cache_path, f"{split}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(data_split, f)
            data_feat[split] = data_split

    test_data = data['test'].dataloader
    del data
    del feature_extractor
    data = data_feat

    for split in parse_splits:
        dataset = DictDataset(data[split])
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=shuffle, num_workers=8, pin_memory=True)
        data[split] = dataloader
    data['test'] = test_data
    print(f"Training Samples: {len(data['train'].dataset)}\tValidation Samples: {len(data['val'].dataset)}\tTest Samples: {len(data['test'].dataset)}")


    model = task_head.to(device)
    steps = len(data['train']) // cfg.TRAINER['gradient_accumulation_steps'] * cfg['TRAINER']['num_train_epochs']
    warmup_steps = len(data['train']) // cfg.TRAINER['gradient_accumulation_steps'] * cfg['TRAINER']['warmup_epochs']
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

    for epoch in trange(training_args.epochs):
        finetune_one_epoch(model, data, loss, epoch, optimizer, scheduler, training_args)
        evaluate_finetune(model, data, loss, epoch, training_args, val_split='val', eval_metric=cfg.DATASET['eval_metric'])
    head_weights = model.state_dict().copy()
    del model
    models = construct_downstream_models(cfg)

    model = models[args.finetune_modal]
    model.head.load_state_dict(head_weights)
    model = model.to(device)

    final_metrics = evaluate_finetune(model, data, loss, epoch+1, training_args, val_split='test', eval_metric=cfg.DATASET['eval_metric'], input_key='image')
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
        torch.save(model.head.state_dict(), os.path.join(cfg['TRAINER']['output_dir'], 'lp_model.pth'))