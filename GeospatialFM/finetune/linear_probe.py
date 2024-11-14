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
from GeospatialFM.data_process.collate_func import modal_specific_collate_fn, linear_probe_collate_fn
from GeospatialFM.finetune.utils import get_loss_fn, get_metric, get_task_model

from datasets.fingerprint import Hasher

logger = get_logger(__name__)

def compute_encoding(batch, model, task_type, modal='optical'):
    optical = batch.get("optical", None)
    radar = batch.get("radar", None)
    optical_channel_wv = batch.get("optical_channel_wv", None)
    radar_channel_wv = batch.get("radar_channel_wv", None)  
    spatial_resolution = batch.get("spatial_resolution", None)  
    labels = batch.get("label", None)
    
    optical = None if optical is None else torch.stack(optical).to(model.device)
    radar = None if radar is None else torch.stack(radar).to(model.device)
    optical_channel_wv = None if optical_channel_wv is None else torch.tensor(optical_channel_wv[0]).unsqueeze(0).to(model.device)
    radar_channel_wv = None if radar_channel_wv is None else torch.tensor(radar_channel_wv[0]).unsqueeze(0).to(model.device)
    spatial_resolution = None if spatial_resolution is None else spatial_resolution[0]
    labels = None if labels is None else torch.tensor(labels)

    with torch.no_grad():    
        outputs = model(optical=optical, radar=radar, optical_channel_wv=optical_channel_wv, radar_channel_wv=radar_channel_wv, spatial_resolution=spatial_resolution)
        
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    else:
        outputs = outputs.last_hidden_state
    
    features = outputs[:, :, 0].cpu()

    return {"features": features, "labels": labels}

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
        
    encoder = model.encoder
    task_head = model.classifier
    encoder.cuda().eval()
    
    # preprocess dataset
    compute_encoding_fn = partial(compute_encoding, model=encoder, task_type=args.task_type, modal=args.modal)
    
    for split, dataset_split in dataset.items():
        if args.regenerate_embeddings:
            dataset_split.cleanup_cache_files() 
        new_fingerprint_for_encoder = Hasher.hash((args.pretrained_model_path, args.modal, args.dataset_name, split, args.scale))
        feature_dataset = dataset_split.map(compute_encoding_fn, batched=True, batch_size=64, new_fingerprint=new_fingerprint_for_encoder)
        feature_dataset.remove_columns(['spatial_resolution'])
        if 'optical' in feature_dataset.column_names: feature_dataset.remove_columns(['optical', 'optical_channel_wv'])
        if 'radar' in feature_dataset.column_names: feature_dataset.remove_columns(['radar', 'radar_channel_wv'])
        feature_dataset.set_format(type='torch')
        dataset[split] = feature_dataset
        
    del encoder
    del model.encoder
    model = task_head

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
        greater_is_better=True,
        logging_strategy="epoch",
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
        data_collator=linear_probe_collate_fn,
        compute_metrics=compute_metrics,  # Add the metrics computation function
        compute_loss_func=custom_loss_function  # Pass the custom loss function
    )
    
    # Train and evaluate
    train_result = trainer.train()
    
    # Save the best model first
    trainer.save_model(os.path.join(args.output_dir, "last_model"))
    
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