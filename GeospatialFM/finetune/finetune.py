import argparse
import os
import logging
from pathlib import Path
from datetime import timedelta
import math

import torch

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

import transformers
from transformers import is_wandb_available
from transformers import get_scheduler
from transformers import TrainingArguments
from datasets import load_dataset

from GeospatialFM.models import SpatialSpectralLowRankViTConfig, SpatialSpectralLowRankViTEncoder
from GeospatialFM.finetune.trainer import MAETrainer
from GeospatialFM.finetune.args import parse_args
from GeospatialFM.finetune.utils import get_task_head, SpatialSpectralLowRankViTWithTaskHead, get_dataloader

logger = get_logger(__name__)

def main(args):    
    # Set up accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=12000))
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
                
    # Initialize model
    model_config = SpatialSpectralLowRankViTConfig(**vars(args))
    encoder = SpatialSpectralLowRankViTEncoder(model_config)
    task_head = get_task_head(args.task_type)
    model = SpatialSpectralLowRankViTWithTaskHead(encoder, task_head)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    model.to(accelerator.device, dtype=weight_dtype)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(),       
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )

    # Load dataset
    dataset, collate_fn, dataloader = get_dataloader(args)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(dataloader['train']) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Create TrainingArguments
    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        per_device_train_batch_size=args.train_batch_size,
        full_determinism=False,
        dispatch_batches=None,
        # deepspeed_plugin=None 
    )
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        optimizers=(optimizer, lr_scheduler),
        accelerator=accelerator, 
        data_collator=collate_fn, 
        train_dataloader=dataloader['train'],
        eval_dataloader=dataloader['val'],
        weight_dtype=weight_dtype,
        modal_mode=args.modal_mode,
        early_stop_steps=args.early_stop_steps
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(f"gfm-{args.dataset_name}", config=vars(args), init_kwargs={"wandb": {"name": args.run_name, "dir": args.wandb_dir}})
    
    trainer.train()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)