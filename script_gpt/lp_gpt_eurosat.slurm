#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdmr-delta-gpu   # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=ft_gpt_ben
#SBATCH --time=09:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=yuxuanw8@illinois.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options

# # Load necessary modules
# module reset
# module load python  # Adjust this as needed to load your Python environment

# Set up environment variables
export ROOT_DIR="."
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0  # Ensure all GPUs are available

# Set parameters
DATASET="eurosat"
WD=0.01
SCALE=1

# 5e-3 8e-3 1e-2 3e-2 5e-2 8e-2 1e-1 3e-1

# Loop through learning rates and fine-tune
for LR in 5e-3 8e-3 1e-2 3e-2 5e-2 8e-2 1e-1 3e-1; do
    srun accelerate launch --num_processes=1 GeospatialFM/finetune/linear_probe.py \
        --data_dir /scratch/bdmr/ywan1/data/geospatial \
        --dataset_name $DATASET \
        --task_type classification \
        --scale $SCALE \
        --modal optical \
        --return_dict \
        --per_device_train_batch_size 1024 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 100 \
        --learning_rate $LR \
        --weight_decay $WD \
        --warmup_steps 0 \
        --warmup_ratio 0.2 \
        --report_to none \
        --save_total_limit 1 \
        --seed 42 \
        --mixed_precision bf16 \
        --dataloader_num_workers 4 \
        --dataloader_pin_memory \
        --output_dir $ROOT_DIR/results_wyx/models \
        --logging_dir $ROOT_DIR/results_wyx/logs \
        --wandb_dir $ROOT_DIR/results_wyx/ \
        --run_name GPT_${DATASET}_lr${LR}_wd${WD}_lp \
        --lr_scheduler_type cosine \
        --crop_size 128 \
        --model_name spectralgpt \

done