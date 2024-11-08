# ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
ROOT_DIR="."
DATASET="fmow"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1

for lr in 1e-5; do
    for adam_wd in 0.005; do
        echo "lr: $lr, adam_wd: $adam_wd"
        accelerate launch --main_process_port=10087 --num_processes 2 GeospatialFM/finetune/finetune.py \
            --data_dir /data-4/common/geospatial \
            --dataset_name $DATASET \
            --task_type classification \
            --scale 2 \
            --modal optical \
            --return_dict \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 32 \
            --num_train_epochs 10 \
            --learning_rate $lr \
            --adam_weight_decay $adam_wd \
            --warmup_steps 30 \
            --report_to wandb \
            --save_total_limit 5 \
            --seed 42 \
            --mixed_precision bf16 \
            --dataloader_num_workers 32 \
            --dataloader_pin_memory \
            --output_dir $ROOT_DIR/results_wyx/models \
            --logging_dir $ROOT_DIR/results_wyx/logs \
            --wandb_dir $ROOT_DIR/results_wyx/ \
            --run_name LESSVIT_b2_d6_${DATASET}_s2_lr${lr}_wd${adam_wd} \
            --lr_scheduler_type cosine \
            --channel_embed_dims_per_head 2 \
            --use_perception_field_mask \
            --pretrained_model_path $ROOT_DIR/results/models/LESSVIT_b2_d6/checkpoint-44000/model.safetensors
    done
done