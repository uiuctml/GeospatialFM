from GeospatialFM.scripts.trainer import MAETrainer
from GeospatialFM.models import MultiModalLowRankViTConfig
from GeospatialFM.models.mae import MultiModalMAEViT
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse
import os
import logging
from logging import get_logger
from pathlib import Path
from datetime import timedelta
from transformers import is_wandb_available
from accelerate import InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
import transformers
import math
from transformers import get_scheduler

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAE model")
    parser.add_argument("--output_dir", type=str, default="output", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Location for log files.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--mixed_precision", type=str, default="float32", choices=["float32", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.")
    parser.add_argument("--report_to", type=str, default="wandb", help="The integration to report the results and logs to. Supported platforms are `tensorboard`, `wandb`, `comet_ml` and `clearml`. Use `all` to report to all integrations.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="If set, deletes the older checkpoints in `output_dir`.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="The scheduler type to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    return parser.parse_args()

def main(args):    
    # Set up accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=12000))
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
    model_config = MultiModalLowRankViTConfig()
    model = MultiModalMAEViT(model_config)
    
    # set model to train mode
    model.train()
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    model.to(accelerator.device, dtype=weight_dtype)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(),       
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )
    
    # TODO: add data loader and process data here
    train_dataset = None
    train_dataloader = None
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    trainer = MAETrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        accelerator=accelerator
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("mm-mae-pretraining", config=vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)