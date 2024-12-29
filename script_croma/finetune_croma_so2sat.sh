ROOT_DIR="."
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET="so2sat"
ATTENTION_RADIUS=640
CHECKPOINT=24600
MOE=0
SCALE=1

WD=0.01

for LR in 1e-3 8e-4 5e-4 3e-4 1e-4 8e-5 5e-5 3e-5; do
    accelerate launch --num_processes=4 GeospatialFM/finetune/finetune.py \
        --data_dir /data-4/common/geospatial \
        --dataset_name $DATASET \
        --task_type classification \
        --scale $SCALE \
        --modal optical \
        --return_dict \
        --per_device_train_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 20 \
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
        --run_name CROMA_${DATASET}_lr${LR}_wd${WD} \
        --lr_scheduler_type cosine \
        --crop_size 120 \
        --model_name croma \
        --train_frac 0.1 \
        --val_frac 0.1

done