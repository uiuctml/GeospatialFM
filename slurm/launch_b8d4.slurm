#!/bin/bash
#SBATCH --job-name="b2_d4"
#SBATCH --output="slurm/geospatial_training.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdmr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00
#SBATCH -e slurm_logs/slurm-%j.err
#SBATCH -o slurm_logs/slurm-%j.out

ROOT_DIR="/u/ehzoahis/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1

DECODER_DEPTH=4
EMBED_DIMS=8

accelerate launch GeospatialFM/scripts/train.py \
    --data_dir $ROOT_DIR/data/geospatial/SSL4EO \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.5 \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name LESSVIT_b${EMBED_DIMS}_d${DECODER_DEPTH} \
    --lr_scheduler cosine \
    --channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_depth $DECODER_DEPTH \
    --use_perception_field_mask \
    --resume_from_checkpoint latest \
    --max_grad_norm 1.0 \
    --proj_drop 0.1 \
    --attn_drop 0.1 \
    --drop_path_rate 0.1 \
    --loss_type mse \
    --modal_mode multi \
    --scale 1 \
    --crop_size 128 