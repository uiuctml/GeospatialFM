ROOT_DIR="/home/user/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# MODEL_NAME="LESSVIT-S"
# EMBED_DIM=384
# NUM_HEADS=6
# MLP_RATIO=4.0

# DECODER_DEPTH=4
# DECODER_EMBED_DIM=512
# DECODER_NUM_HEADS=16

# CHANNEL_EMBED_DIMS_PER_HEAD=2
# DECODER_CHANNEL_EMBED_DIMS_PER_HEAD=2
# RANK=1

# accelerate launch --num_processes 4 GeospatialFM/scripts/train.py \
#     --dataset_name enmap \
#     --data_dir /datasets/disk2/geospatial/enmap/enmap \
#     --per_device_train_batch_size 200 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 100 \
#     --learning_rate 1e-4 \
#     --weight_decay 0.05 \
#     --mask_ratio 0.75 \
#     --channel_mask_ratio 0.75 \
#     --warmup_ratio 0.05 \
#     --report_to none \
#     --save_steps 0.1 \
#     --save_total_limit 5 \
#     --seed 42 \
#     --mixed_precision bf16 \
#     --dataloader_num_workers 16 \
#     --dataloader_pin_memory \
#     --output_dir ./results_wyx/models \
#     --logging_dir ./results_wyx/logs \
#     --wandb_dir ./results_wyx/results/ \
#     --run_name ${MODEL_NAME}_b${EMBED_DIM}_d${DECODER_DEPTH}_r${RANK} \
#     --lr_scheduler_type cosine \
#     --embed_dim $EMBED_DIM \
#     --num_heads $NUM_HEADS \
#     --mlp_ratio $MLP_RATIO \
#     --channel_embed_dims_per_head $CHANNEL_EMBED_DIMS_PER_HEAD \
#     --decoder_embed_dim $DECODER_EMBED_DIM \
#     --decoder_depth $DECODER_DEPTH \
#     --decoder_num_heads $DECODER_NUM_HEADS \
#     --decoder_channel_embed_dims_per_head $DECODER_CHANNEL_EMBED_DIMS_PER_HEAD \
#     --decoder_out_chans 202 \
#     --use_perception_field_mask \
#     --max_grad_norm 1.0 \
#     --proj_drop 0.1 \
#     --attn_drop 0.1 \
#     --drop_path_rate 0.1 \
#     --loss_type mse \
#     --modal_mode optical \
#     --scale 1 \
#     --crop_size 128 \
#     --init_values 1.0 \
#     --rank $RANK \
#     --use_rope_embed \
#     --channel_dropout 0.7 0.8 \
#     --resume_from_checkpoint /home/yuxuanwan/GeospatialFM/results_wyx/models/LESSVIT-S_b384_d4_r1/checkpoint-24960


DECODER_DEPTH=8
EMBED_DIMS=4
RANK=1

accelerate launch GeospatialFM/scripts/train.py \
    --dataset_name enmap \
    --data_dir /datasets/disk2/geospatial/enmap/enmap \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 200 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.75 \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory \
    --output_dir ./results_wyx/models \
    --logging_dir ./results_wyx/logs \
    --wandb_dir ./results_wyx/results/ \
    --run_name LESSVIT_b${EMBED_DIMS}_d${DECODER_DEPTH}_r${RANK}_v2 \
    --lr_scheduler cosine \
    --channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_depth $DECODER_DEPTH \
    --decoder_out_chans 202 \
    --use_perception_field_mask \
    --attention_radius 360 \
    --max_grad_norm 1.0 \
    --proj_drop 0.1 \
    --attn_drop 0.1 \
    --drop_path_rate 0.1 \
    --loss_type mse \
    --modal_mode optical \
    --scale 1 \
    --crop_size 128 \
    --patch_size 4 \
    --init_values 1.0 \
    --rank $RANK \
    --use_rope_embed \
    --channel_dropout 0.7 0.8 \
    --crop_size 32 \
