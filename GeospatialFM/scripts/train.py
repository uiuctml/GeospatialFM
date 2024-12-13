import os
import logging
import math
from functools import partial

from accelerate.logging import get_logger
from transformers import is_wandb_available
from transformers import TrainingArguments

from GeospatialFM.datasets.ssl4eo import get_ssl4eo_metadata, SSL4EODataset
from GeospatialFM.data_process import pretrain_transform, multimodal_collate_fn
from GeospatialFM.models import SpatialSpectralLowRankViTConfig, SpatialSpectralMAEViT
from GeospatialFM.scripts.trainer import MAETrainer
from GeospatialFM.scripts.args import parse_args
from GeospatialFM.scripts.utils import calculate_modal_loss, get_lasted_checkpoint

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
                
    # Initialize model
    model_config = SpatialSpectralLowRankViTConfig(**vars(args))
    model = SpatialSpectralMAEViT(model_config)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    metadata = get_ssl4eo_metadata()
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    radar_mean, radar_std = metadata["s1"]["mean"], metadata["s1"]["std"]
    
    transform = partial(pretrain_transform, optical_mean=optical_mean, optical_std=optical_std, radar_mean=radar_mean, radar_std=radar_std)
    collate_fn = partial(multimodal_collate_fn, transform=transform, random_crop=args.random_crop, scale=args.scale, crop_size=args.crop_size)
    
    dataset = dict(train=SSL4EODataset(root=args.data_dir))

    custom_loss_function = partial(calculate_modal_loss, loss_type=args.loss_type)
    
    if args.resume_from_checkpoint == "latest":
        args.resume_from_checkpoint = get_lasted_checkpoint(args)
        print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
    
    training_args = TrainingArguments(
        **{k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__},
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        logging_strategy="steps",
        logging_steps=1,
        ddp_find_unused_parameters=False,
    )
    
    # Set up wandb first if using it
    if args.report_to == "wandb" :
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if training_args.local_rank == 0:
            wandb.init(
                project=f"less-pretrain",
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args)
            )
    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=collate_fn,
        compute_loss_func=custom_loss_function,
        modal_mode=args.modal_mode,
    )
    
    total_batch_size = trainer.args.per_device_train_batch_size * trainer.accelerator.num_processes * trainer.args.gradient_accumulation_steps
    max_steps = trainer.args.max_steps if trainer.args.max_steps != -1 else math.ceil(len(trainer.train_dataset) / total_batch_size) * trainer.args.num_train_epochs

    if training_args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(trainer.train_dataset)}")
        logger.info(f"  Num Epochs = {trainer.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {trainer.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {trainer.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)