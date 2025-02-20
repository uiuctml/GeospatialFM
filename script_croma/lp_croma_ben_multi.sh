ROOT_DIR="."
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=2

DATASET="bigearthnet"
ATTENTION_RADIUS=640
CHECKPOINT=24600
MOE=0
SCALE=1

WD=0.01
# 5e-3 8e-3 1e-2 3e-2 5e-2 8e-2 1e-1 3e-1
# 9e-1 8e-1 7e-1 6e-1 5e-1 4e-1 3e-1 2e-1 1e-1 9e-2 8e-2 7e-2 6e-2 5e-2 4e-2 3e-2 2e-2 1e-2 9e-3 8e-3 7e-3 6e-3 5e-3 4e-3 3e-3 2e-3 1e-3 9e-4 8e-4 7e-4 6e-4 5e-4 4e-4 3e-4 2e-4 1e-4
for LR in 5e-3 8e-3 1e-2 3e-2 5e-2 8e-2 1e-1 3e-1; do
    accelerate launch GeospatialFM/finetune/linear_probe.py \
        --data_dir /data-4/common/geospatial \
        --dataset_name $DATASET \
        --task_type multilabel \
        --scale $SCALE \
        --modal multi \
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
        --dataloader_num_workers 32 \
        --dataloader_pin_memory \
        --output_dir $ROOT_DIR/results_wyx/models \
        --logging_dir $ROOT_DIR/results_wyx/logs \
        --wandb_dir $ROOT_DIR/results_wyx/ \
        --run_name CROMA_${DATASET}_lr${LR}_wd${WD}_lp_multi \
        --lr_scheduler_type cosine \
        --crop_size 120 \
        --model_name croma \
        --train_frac 0.1 \
        --val_frac 0.1 \

done