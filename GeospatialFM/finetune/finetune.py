import argparse
import os
import logging
from pathlib import Path
from datetime import timedelta
import math

import torch
from accelerate.logging import get_logger

import transformers
from transformers import is_wandb_available
from transformers import TrainingArguments, Trainer
from typing import Dict
import numpy as np

from functools import partial

from GeospatialFM.finetune.args import parse_args
from GeospatialFM.datasets.GFMBench.utils import get_dataset, get_metadata
from GeospatialFM.data_process.transforms import get_transform
from GeospatialFM.data_process.collate_func import modal_specific_collate_fn, apply_normalization
from GeospatialFM.finetune.utils import get_loss_fn, get_metric, get_task_model

logger = get_logger(__name__)

def main(args):    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    metadata = get_metadata(args.dataset_name)
    args.crop_size = metadata["size"] if args.crop_size is None else args.crop_size
    
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    radar_mean, radar_std = metadata["s1"]["mean"], metadata["s1"]["std"]
    
    collate_fn = partial(modal_specific_collate_fn, modal=args.modal)
    
    train_transform, eval_transform = get_transform(args.task_type, args.crop_size, args.scale, args.random_rotation, 
                                                    optical_mean, optical_std, radar_mean, radar_std)
    dataset = get_dataset(args, train_transform, eval_transform)
    
    # Initialize model
    model = get_task_model(args, metadata["num_classes"], metadata["size"])
    # load from checkpoint if provided
    if args.pretrained_model_path:
        from safetensors import safe_open
        with safe_open(args.pretrained_model_path, framework="pt", device="cpu") as f:
            # Load only encoder weights
            for key in f.keys():
                if key.startswith("encoder."):
                    # Get the corresponding key in target model
                    param = f.get_tensor(key)
                    model.state_dict()[key].copy_(param)
    
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # get loss function and metric
    custom_loss_function = get_loss_fn(args.task_type)
    compute_metrics, metric_name = get_metric(args.task_type)

    # Create TrainingArguments with evaluation settings
    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        full_determinism=False,
        dispatch_batches=None,
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        load_best_model_at_end=True,  
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        metric_for_best_model=metric_name
    )
    
    # Set up wandb first if using it
    if args.report_to == "wandb" :
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if training_args.local_rank == 0:
            wandb.init(
                project=f"gfm-{args.dataset_name}",
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args)
            )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,  # Add the metrics computation function
        compute_loss_func=custom_loss_function  # Pass the custom loss function
    )
    
    # Train and evaluate
    train_result = trainer.train()
    
    # Save the best model first
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    
    # Final evaluation
    metrics = trainer.evaluate(eval_dataset=dataset['test'])
    
    # Log the metrics
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    
    # Final evaluation
    metrics = trainer.evaluate(eval_dataset=dataset['val'])
    
    # Log the metrics
    trainer.log_metrics("val", metrics)
    trainer.save_metrics("val", metrics)
    
    # # Save the final model
    # trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    # Save training state
    trainer.save_state()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)