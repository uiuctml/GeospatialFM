ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET="eurosat"
EMBED_DIMS=1
DEPTH=4
ATTENTION_RADIUS=640
CHECKPOINT=24600
MOE=0
SCALE=2

accelerate launch GeospatialFM/finetune/hp_search.py \
    --data_dir $ROOT_DIR/data/geospatial/ \
    --dataset_name $DATASET \
    --task_type classification \
    --scale $SCALE \
    --modal optical \
    --return_dict \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.01 \
    --warmup_steps 0 \
    --warmup_ratio 0.2 \
    --report_to none \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 32 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name LESSVIT_b${EMBED_DIMS}_d${DEPTH}_${DATASET}_ckpt${CHECKPOINT}_moe${MOE}_scale${SCALE} \
    --lr_scheduler_type cosine \
    --channel_embed_dims_per_head $EMBED_DIMS \
    --use_perception_field_mask \
    --pretrained_model_path $ROOT_DIR/results/models/LESSVIT_b${EMBED_DIMS}_d${DEPTH}/checkpoint-${CHECKPOINT}/model.safetensors \
    --use_moe \
    --num_experts $MOE \
    --attention_radius $ATTENTION_RADIUS
