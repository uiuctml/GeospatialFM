ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0

DATASET="eurosat"
EMBED_DIMS=1
DEPTH=8
ckpt=70000

for lr in 5e-3 6e-3 7e-3 8e-3 9e-3; do
    for adam_wd in 0; do
        echo "lr: $lr, adam_wd: $adam_wd"
        accelerate launch --config_file ~/.cache/huggingface/accelerate/lp_config.yaml \
        GeospatialFM/finetune/linear_probe.py \
        --data_dir $ROOT_DIR/data/geospatial \
        --dataset_name $DATASET \
        --task_type classification \
        --scale 4 \
        --modal optical \
        --return_dict \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 1024 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 100 \
        --learning_rate $lr \
        --adam_weight_decay $adam_wd \
        --warmup_ratio 0.2 \
        --warmup_steps 0 \
        --report_to wandb \
        --save_strategy no \
        --seed 42 \
        --mixed_precision bf16 \
        --dataloader_num_workers 32 \
        --dataloader_pin_memory \
        --output_dir $ROOT_DIR/results/models \
        --logging_dir $ROOT_DIR/results/logs \
        --wandb_dir $ROOT_DIR/results/ \
        --run_name LESSVIT_b${EMBED_DIMS}_d${DEPTH}_${DATASET}_lp_lr${lr}_wd${adam_wd}_ckpt${ckpt}_moe0 \
        --lr_scheduler_type cosine \
        --channel_embed_dims_per_head $EMBED_DIMS \
        --use_perception_field_mask \
        --pretrained_model_path $ROOT_DIR/results/models/LESSVIT_b${EMBED_DIMS}_d${DEPTH}/checkpoint-${ckpt}/model.safetensors \
        --regenerate_embeddings
    done
done