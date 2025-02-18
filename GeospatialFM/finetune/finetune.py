import argparse
import os
import logging
from pathlib import Path
from datetime import timedelta
import math
import optuna

import torch
from accelerate.logging import get_logger

import transformers
from transformers import is_wandb_available
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets.fingerprint import Hasher
from typing import Dict
import numpy as np
import json

from functools import partial

from GeospatialFM.finetune.args import parse_args
from GeospatialFM.datasets.GFMBench.utils import get_dataset, get_metadata
from GeospatialFM.data_process.transforms import get_transform
from GeospatialFM.data_process.collate_func import modal_specific_collate_fn, linear_probe_collate_fn
from GeospatialFM.finetune.utils import get_loss_fn, get_metric, get_task_model

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

def model_init_template(trial, lp=False):
    args = parse_args()
    metadata = get_metadata(args.dataset_name, args.dataset_version)
    # Initialize model
    model = get_task_model(args, metadata["num_classes"], args.crop_size)
    # load from checkpoint if provided
    if args.pretrained_model_path:
        model.load_pretrained_encoder(args.pretrained_model_path)
        
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    if lp:
        return model.classifier
    else:
        return model

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3]),
    }

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
        
    if args.logging_dir is not None:
        os.makedirs(args.logging_dir, exist_ok=True)

    # Load dataset
    metadata = get_metadata(args.dataset_name, args.dataset_version)
    args.crop_size = metadata["size"] if args.crop_size is None else args.crop_size
    
    try: # for sentinel-2 and sentinel-1
        optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
        radar_mean, radar_std = metadata["s1"]["mean"], metadata["s1"]["std"]
    except: # for landsat and other datasets
        optical_mean, optical_std = metadata['mean'], metadata['std']
        radar_mean, radar_std = None, None
        
    train_transform, eval_transform = get_transform(args.task_type, args.crop_size, args.scale, args.random_rotation, 
                                                    optical_mean, optical_std, radar_mean, radar_std, args.dataset_name)
    
    dataset = get_dataset(args, train_transform, eval_transform)
    
    if args.lp:
        model = model_init_template(None, lp=False)
        
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        encoder = model.encoder
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
        model_init = partial(model_init_template, lp=True)
        collate_fn = linear_probe_collate_fn
    else:
        model_init = partial(model_init_template, lp=False)
        collate_fn = partial(modal_specific_collate_fn, modal=args.modal)
    
    # get loss function and metric
    custom_loss_function = get_loss_fn(args.task_type)
    compute_metrics, metric_name = get_metric(args.task_type, metadata["num_classes"])

    # Create TrainingArguments with evaluation settings
    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        full_determinism=False,
        dispatch_batches=None,
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        load_best_model_at_end=True if not args.lp else False,
        greater_is_better=True,
        logging_strategy="steps" if not args.lp else "epoch",
        logging_steps=1 if not args.lp else None,
        logging_first_step=True,
        metric_for_best_model=metric_name
    )
    
    callbacks = []
    if args.use_early_stopping and not args.lp:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold))
    
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
    
    # if args.use_optuna:
    if args.use_optuna:
        # assert single GPU
        assert training_args.n_gpu == 1, "Optuna is not supported for multi-GPU training"
        trainer = Trainer(
            model = None,
            model_init=model_init,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            data_collator=collate_fn,
            compute_metrics=compute_metrics,  # Add the metrics computation function
            compute_loss_func=custom_loss_function,  # Pass the custom loss function
            callbacks=callbacks
        )
    
        # Train and evaluate
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=args.n_trials,
            storage=f"sqlite:///{args.logging_dir}/finetune.db",
            study_name=args.run_name,
            load_if_exists=True,
            # pruner=optuna.pruners.NopPruner()  # FIXME: Pruner is not compatible with our server now, fix it later
        )
    
        # Print the best hyperparameters and their performance
        logger.info(f"\n\nBest trial:")
        logger.info(f"Value (objective): {best_trial.objective}")
        logger.info("Parameters:")
        for key, value in best_trial.hyperparameters.items():
            logger.info(f"\t{key}: {value}")
                
        # write the best trial to a json file
        best_trial_dict = {
            "objective": best_trial.objective,
            "parameters": best_trial.hyperparameters
        }
        with open(os.path.join(args.logging_dir, f"{args.run_name}.json"), "w") as f:
            json.dump(best_trial_dict, f)
            
        # modify the training args
        for key, value in best_trial.hyperparameters.items():
            if key in training_args.__dict__:
                setattr(training_args, key, value)
            
        # Load the best model
        trainer = Trainer(
            model=model_init(best_trial),  # Initialize model with best hyperparameters
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            compute_loss_func=custom_loss_function
        )
        
    else:
        trainer = Trainer(
            model=model_init(None),
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            compute_loss_func=custom_loss_function
        )
    
    # Train the model with best hyperparameters
    train_result = trainer.train()
    
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