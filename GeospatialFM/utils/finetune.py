import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from collections import defaultdict
try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .precision import get_autocast
from GeospatialFM.loss import *
from .metrics import *

def finetune_one_epoch(model, data, loss, epoch, optimizer, scheduler, args, input_keyword='image', target_keyword='label'):
    assert input_keyword in ['image', 'radar', 'feature', 'images'] # CHANGE
    assert target_keyword in ['label', 'mask'] # CHANGE

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    model.train()

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

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
        if input_keyword == 'images': # CHANGE
            image_1 = batch['image1'].to(device=device, non_blocking=True)
            image_2 = batch['image2'].to(device=device, non_blocking=True)
            model_input = (image_1, image_2)
        else:
            model_input = batch[input_keyword].to(device=device, non_blocking=True)

        label = batch[target_keyword].to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        
        # if args.accum_freq == 1:
        with autocast():
            model_out = model(model_input)
            losses = loss(model_out, label, output_dict=True)

            total_loss = sum(losses.values())
            losses["loss"] = total_loss

        total_loss.backward()

        if (i + 1) % args.accum_freq == 0:
            optimizer.step()
            optimizer.zero_grad() 

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(label) # CHANGE
            num_samples = batch_count * batch_size * args.accum_freq
            samples_per_epoch = dataloader.num_samples
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
            if args.print_log:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
                )

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

def evaluate_finetune(model, data, loss, epoch, args, val_split='val', eval_metric='accuracy', input_keyword='image', target_keyword='label'):
    assert input_keyword in ['image', 'radar', 'feature', 'images'] # CHANGE
    assert target_keyword in ['label', 'mask'] # CHANGE
    assert eval_metric in ['accuracy', 'mAP', 'f1'] # CHANGE
    eval_fn = get_eval_fn(eval_metric)
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision)
    # input_dtype = get_input_dtype(args.precision)

    if val_split in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # all_preds, all_labels = [], []
        dataloader = data[val_split].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_losses = defaultdict(float)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if input_keyword == 'images': # CHANGE
                    image_1 = batch['image1'].to(device=device, non_blocking=True)
                    image_2 = batch['image2'].to(device=device, non_blocking=True)
                    model_input = (image_1, image_2)
                else:
                    model_input = batch[input_keyword].to(device=device, non_blocking=True)

                label = batch[target_keyword].to(device=device, non_blocking=True)
                batch_size = len(label)

                with autocast():
                    model_out = model(model_input)
                    losses = loss(model_out, label, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                    
                    image_metric = eval_fn(model_out, label, return_dict=True)
                    losses.update(image_metric)

                num_samples += batch_size

                for key, val in losses.items():
                    try:
                        cumulative_losses[key] += val.item() * batch_size
                    except:
                        cumulative_losses[key] += val * batch_size

                if is_master(args) and (i % 100) == 0:
                    loss_log = f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]" 
                    loss_log += " ".join(
                        [
                            f"{key.replace('_', ' ')}: {val / num_samples:.6f}" 
                            for key, val in cumulative_losses.items()
                        ]
                    )
                    logging.info(loss_log)

            for key, val in cumulative_losses.items():
                metrics[key] = val / num_samples

            metrics.update({"epoch": epoch, "num_samples": num_samples})

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
        step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics