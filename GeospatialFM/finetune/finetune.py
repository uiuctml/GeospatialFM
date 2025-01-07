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
from typing import Dict
import numpy as np
import json

from functools import partial

from GeospatialFM.finetune.args import parse_args
from GeospatialFM.datasets.GFMBench.utils import get_dataset, get_metadata, get_baseline_metadata
from GeospatialFM.data_process.transforms import get_transform
from GeospatialFM.data_process.collate_func import modal_specific_collate_fn
from GeospatialFM.finetune.utils import get_loss_fn, get_metric, get_task_model, get_baseline_model

logger = get_logger(__name__)

def model_init(trial):
    args = parse_args()
    metadata = get_metadata(args.dataset_name)
    
    if args.model_name:
        model = get_baseline_model(args, metadata["num_classes"], metadata["size"])
        model.load_pretrained_encoder(args.pretrained_model_path)
        return model

    # Initialize model
    image_size = args.crop_size if args.dataset_name.lower().strip() == "landsat" and args.crop_size is not None else metadata['size']
    model = get_task_model(args, metadata["num_classes"], image_size)
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
    return model

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.05, 0.1, 0.2]),
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
    metadata = get_metadata(args.dataset_name)
    args.crop_size = metadata["size"] if args.crop_size is None else args.crop_size
    
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"] if args.dataset_name.lower().strip() != "landsat" else (metadata['mean'], metadata['std'])
    radar_mean, radar_std = metadata["s1"]["mean"], metadata["s1"]["std"] if args.dataset_name.lower().strip() != "landsat" else (None, None)
    data_bands = metadata["s2c"]["bands"] if args.dataset_name.lower().strip() != "landsat" else metadata["bands"]
    
    collate_fn = partial(modal_specific_collate_fn, modal=args.modal)
    
    train_transform, eval_transform = get_transform(args.task_type, args.crop_size, args.scale, args.random_rotation, 
                                                    optical_mean, optical_std, radar_mean, radar_std, 
                                                    data_bands=data_bands, model_bands=get_baseline_metadata(args), args.dataset_name)
    dataset = get_dataset(args, train_transform, eval_transform)
    
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
        load_best_model_at_end=True,  
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        metric_for_best_model=metric_name
    )
    
    callbacks = []
    if args.use_early_stopping:
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
            pruner=optuna.pruners.NopPruner()  # FIXME: Pruner is not compatible with our server now, fix it later
        )
    
        if training_args.local_rank == 0:
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

    else:
        trainer = Trainer(
        model=model_init(None),  # Initialize model with best hyperparameters
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