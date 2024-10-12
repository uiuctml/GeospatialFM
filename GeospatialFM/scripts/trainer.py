import os
import torch
import math
from transformers import Trainer, get_scheduler
from typing import Dict
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedType
import logging
from collections import defaultdict
import shutil
from torch.utils.data import DataLoader
import numpy as np
logger = logging.getLogger(__name__)

class MAETrainer(Trainer):
    def __init__(self, model, args, train_dataset, optimizers, data_collator, accelerator, weight_dtype, train_dataloader=None, modal_mode=None, early_stop_steps=None):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            optimizers=optimizers,
            data_collator=data_collator
        )
        self.accelerator = accelerator
        self.args = args
        self.global_step = 0
        self.initial_global_step = 0
        self.first_epoch = 0
        self.weight_dtype = weight_dtype
        self.modal_mode = modal_mode
        self.max_grad_norm = args.max_grad_norm  # Add this line
        self.early_stop_steps = early_stop_steps
        self.train_dataloader = train_dataloader

    def compute_loss(self, model, inputs, return_outputs=False, mask_ratio=None, channel_mask_ratio=None):
        optical = inputs.get("optical").to(self.accelerator.device, dtype=self.weight_dtype)
        radar = inputs.get("radar").to(self.accelerator.device, dtype=self.weight_dtype)
        optical_channel_wv = inputs.get("optical_channel_wv")
        radar_channel_wv = inputs.get("radar_channel_wv")
        spatial_resolution = inputs.get("spatial_resolution")
        
        if self.modal_mode == "random":
            modal = np.random.choice(['multi', 'optical', 'radar'])
        else:
            modal = self.modal_mode
        
        outputs = model(
            optical=optical,
            radar=radar,
            optical_channel_wv=optical_channel_wv,
            radar_channel_wv=radar_channel_wv,
            mask_ratio=mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
            spatial_resolution=spatial_resolution,
            modal=modal
        )

        loss = self.calculate_l1_loss(outputs)

        return (loss, outputs) if return_outputs else loss

    def calculate_mse_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = {}
        total_loss = 0.
        target = outputs['target']
        for modal in ['optical', 'radar', 'multi']:
            if f'{modal}_recon' in outputs:
                recon = outputs[f'{modal}_recon']
                channel_mask = outputs[f'{modal}_channel_mask']
                pos_mask = outputs[f'{modal}_pos_mask']
                # positional MSE
                pos_loss = (torch.mean((recon - target) ** 2, dim=[1, 3]) * pos_mask).sum() / pos_mask.sum()
                # channel MSE
                channel_loss = (torch.mean((recon - target) ** 2, dim=[2, 3]) * channel_mask).sum() / channel_mask.sum()
                loss[f"{modal}_pos_loss"] = pos_loss
                loss[f"{modal}_channel_loss"] = channel_loss
                loss[f"{modal}_loss"] = pos_loss + channel_loss
                total_loss += loss[f"{modal}_loss"]
        loss['total_loss'] = total_loss
        return loss
    
    def calculate_l1_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = {}
        total_loss = 0.
        target = outputs['target']
        for modal in ['optical', 'radar', 'multi']:
            if f'{modal}_recon' in outputs:
                recon = outputs[f'{modal}_recon']
                channel_mask = outputs[f'{modal}_channel_mask'] # B, C
                pos_mask = outputs[f'{modal}_pos_mask']
                # positional MSE
                pos_loss = torch.mean((recon - target).abs(), dim=[3]) # B, C, HW
                if channel_mask.sum() > 0:
                    pos_loss = (pos_loss * (1 - channel_mask).unsqueeze(-1)).sum(dim=1) / (1 - channel_mask).sum(dim=1, keepdim=True) # B, HW
                else:
                    pos_loss = pos_loss.mean(dim=1)  # Average over channels if no channel masking
                
                if pos_mask.sum() == 0:
                    # pos_loss = pos_loss.mean()
                    pos_loss = torch.zeros_like(pos_loss.mean())
                else:
                    pos_loss = (pos_loss * pos_mask).sum() / pos_mask.sum()
                # channel MSE
                channel_loss = torch.mean((recon - target).abs(), dim=[2, 3])
                if channel_mask.sum() == 0:
                    # channel_loss = channel_loss.mean()
                    channel_loss = torch.zeros_like(channel_loss.mean())
                else:
                    channel_loss = (channel_loss * channel_mask).sum() / channel_mask.sum()
                loss[f"{modal}_pos_loss"] = pos_loss
                loss[f"{modal}_channel_loss"] = channel_loss
                loss[f"{modal}_loss"] = pos_loss + channel_loss
                total_loss += loss[f"{modal}_loss"]
        loss['total_loss'] = total_loss
        return loss
    
    def get_train_dataloader(self):
        if self.train_dataloader is not None:
            return self.train_dataloader
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        data_collator = self.data_collator
        train_dataset = self.train_dataset
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True
        )

    def train(self):
        # Prepare everything
        model, optimizer, self.train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.get_train_dataloader(), self.lr_scheduler
        )

        # Recalculate training steps
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / self.num_update_steps_per_epoch)

        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        else:
            self.initial_global_step = 0
            
        if self.early_stop_steps is not None:
            self.args.max_train_steps = self.early_stop_steps

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=self.initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
        
        for epoch in range(self.first_epoch, self.args.num_train_epochs):
            train_losses = defaultdict(float)
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(model):
                    loss = self.compute_loss(model, batch)
                    
                    for key, value in loss.items():
                        avg_loss = self.accelerator.gather(value.repeat(self.args.train_batch_size)).mean()
                        train_losses[key] += avg_loss.item() / self.args.gradient_accumulation_steps
                    
                    self.accelerator.backward(loss['total_loss'])

                    # Add gradient clipping here
                    if self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    self.accelerator.log(train_losses | dict(epoch=epoch, lr=lr_scheduler.get_last_lr()[0]), step=self.global_step)
                    train_losses = defaultdict(float)

                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint()

                logs = {"step_loss": loss['total_loss'].detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if self.global_step >= self.args.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()
        self.save_checkpoint()
        self.accelerator.end_training()

    def save_checkpoint(self):
        if self.accelerator.is_main_process or self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

            if self.args.save_total_limit is not None:
                self.rotate_checkpoints()

    def rotate_checkpoints(self):
        checkpoints = [f for f in os.listdir(self.args.output_dir) if f.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        
        # If we exceed the limit, remove the oldest checkpoints
        if len(checkpoints) > self.args.save_total_limit:
            num_to_remove = len(checkpoints) - self.args.save_total_limit
            removing_checkpoints = checkpoints[:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    def load_checkpoint(self):
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            self.initial_global_step = 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.args.output_dir, path))
            self.global_step = int(path.split("-")[1])
            self.initial_global_step = self.global_step
            self.first_epoch = self.global_step // self.num_update_steps_per_epoch