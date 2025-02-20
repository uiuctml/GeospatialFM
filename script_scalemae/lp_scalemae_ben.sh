# Set up environment variables
export ROOT_DIR="."
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0  # Ensure all GPUs are available

# Set parameters
DATASET="bigearthnet"
WD=0.01
SCALE=1

# 5e-3 8e-3 1e-2 3e-2 5e-2 8e-2 1e-1 3e-1

# Loop through learning rates and fine-tune
for LR in 5e-3 8e-3 1e-2 3e-2 5e-2 8e-2 1e-1 3e-1; do
    accelerate launch --num_processes=1 GeospatialFM/finetune/linear_probe.py \
        --data_dir /data-4/common/geospatial \
        --dataset_name $DATASET \
        --task_type multilabel \
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
        --run_name ScaleMAE_${DATASET}_lr${LR}_wd${WD}_lp \
        --lr_scheduler_type cosine \
        --crop_size 224 \
        --model_name scalemae \
        --embed_dim 1024 \
        --train_frac 0.1 \
        --val_frac 0.1

done