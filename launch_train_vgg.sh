ROOT_DIR="."
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET="bigearthnet"
TASK_TYPE="multilabel"
LR=1e-4

PORT=10087
CROP_SIZE=120

accelerate launch --main_process_port $PORT --num_processes 1 GeospatialFM/finetune/finetune_vgg.py \
    --data_dir /home/jovyan/workspace/yuxuanwan/data \
    --dataset_name $DATASET \
    --task_type $TASK_TYPE \
    --modal multi \
    --return_dict \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate $LR \
    --weight_decay 0.01 \
    --warmup_steps 0 \
    --warmup_ratio 0.2 \
    --report_to none \
    --save_total_limit 1 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name VGG19_${DATASET} \
    --lr_scheduler_type cosine \
    --crop_size $CROP_SIZE