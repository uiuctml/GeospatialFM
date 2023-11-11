import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
# from torch.nn.parallel.distributed import DistributedDataParallel
from collections import defaultdict
try:
    import wandb
except ImportError:
    wandb = None

# from open_clip import get_input_dtype, CLIP, CustomTextCLIP
# from .distributed import is_master
# from .zero_shot import zero_shot_eval
from .precision import get_autocast
from GeospatialFM.loss import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def train_one_epoch(model, data, loss, epoch, optimizer, scheduler, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    model.train()

    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_radar, accum_label, accum_features = [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        images, radar, label = batch['image'], batch['radar'], batch['label']
        images = images.to(device=device, non_blocking=True)
        radar = radar.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)  

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, radar, args.mask_ratio)
                logit_scale = model_out.get("logit_scale")
                model_out['labels'] = label
                if isinstance(loss, list):
                    losses = {}
                    for l in loss:
                        losses.update(l(**model_out, output_dict=True))
                else:
                    losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            total_loss.backward()
        else:
            raise NotImplementedError
            # TODO: combine MAE with CLIP loss
            # First, cache the features without any gradient tracking.
            # with torch.no_grad():
            #     with autocast():
            #         model_out = model(images, radar)

            #         for f in ("logit_scale", "logit_bias"):
            #             model_out.pop(f, None)

            #         for key, val in model_out.items():
            #             if key in accum_features:
            #                 accum_features[key].append(val)
            #             else:
            #                 accum_features[key] = [val]

            #     accum_images.append(images)
            #     accum_radar.append(radar)
            #     accum_label.append(label)


            # # If (i + 1) % accum_freq is not zero, move on to the next batch.
            # if ((i + 1) % args.accum_freq) > 0:
            #     # FIXME this makes data time logging unreliable when accumulating
            #     continue

            # # Now, ready to take gradients for the last accum_freq batches.
            # # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # # Call backwards each time, but only step optimizer at the end.
            # optimizer.zero_grad()
            # for j in range(args.accum_freq):
            #     images = accum_images[j]
            #     radar = accum_radar[j]
            #     with autocast():
            #         model_out = model(images, radar)

            #         inputs_no_accum = {}
            #         inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
            #         if "logit_bias" in model_out:
            #             inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

            #         inputs = {}
            #         for key, val in accum_features.items():
            #             accumulated = accum_features[key]
            #             inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

            #         losses = loss(**inputs, **inputs_no_accum, output_dict=True)
            #         del inputs
            #         del inputs_no_accum
            #         total_loss = sum(losses.values())
            #         losses["loss"] = total_loss

            #     total_loss.backward()

        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        optimizer.step()
        scheduler.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_radar, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            print(logit_scale)
            # logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
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

def evaluate(model, data, loss, epoch, args, val_split='val'):
    metrics = {}
    # if not is_master(args):
    #     return metrics
    device = torch.device(args.device)
    model.eval()

    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    # input_dtype = get_input_dtype(args.precision)

    if val_split in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data[val_split].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_losses = defaultdict(float)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, radar, gt_label = batch['image'], batch['radar'], batch['label']
                images = images.to(device=device, non_blocking=True)
                radar = radar.to(device=device, non_blocking=True)
                gt_label = gt_label.to(device=device, non_blocking=True)

                batch_size = len(images)

                with autocast():
                    model_out = model(images, radar, args.mask_ratio)
                    model_out['labels'] = gt_label
                    if isinstance(loss, list):
                        losses = {}
                        for l in loss:
                            losses.update(l(**model_out, output_dict=True))
                    else:
                        losses = loss(**model_out, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                    # calculate accuracy
                    optical_logits = model_out.get('optical_logits')
                    radar_logits = model_out.get('radar_logits')
                    if optical_logits is not None and radar_logits is not None:
                        image_preds = torch.argmax(model_out['optical_logits'], dim=1)
                        image_acc = (image_preds == gt_label).sum().item() / len(gt_label)
                        losses['image_acc'] = image_acc
                        radar_preds = torch.argmax(model_out['radar_logits'], dim=1)
                        radar_acc = (radar_preds == gt_label).sum().item() / len(gt_label)
                        losses['radar_acc'] = radar_acc

                for key, val in losses.items():
                    try:
                        cumulative_losses[key] += val.item() * batch_size
                    except:
                        cumulative_losses[key] += val * batch_size

                num_samples += batch_size
                if (i % 100) == 0:
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


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
