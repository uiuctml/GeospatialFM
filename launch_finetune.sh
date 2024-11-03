ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch GeospatialFM/finetune/finetune.py \
    --data_dir $ROOT_DIR/data/geospatial \
    --dataset_name eurosat \
    --task_type classification \
    --scale 2 \
    --modal optical \
    --return_dict \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --learning_rate 3e-4 \
    --adam_weight_decay 0.05 \
    --warmup_steps 30 \
    --report_to wandb \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name LESSVIT_b2_d6_eurosat_s2 \
    --lr_scheduler_type cosine \
    --channel_embed_dims_per_head 2 \
    --use_perception_field_mask \
    --pretrained_model_path $ROOT_DIR/results/models/LESSVIT_b2_d6/checkpoint-33000/model.safetensors