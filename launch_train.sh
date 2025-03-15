ROOT_DIR="/home/user/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1

DECODER_DEPTH=4
EMBED_DIMS=2
RANK=8

accelerate launch GeospatialFM/scripts/train.py \
    --data_dir $ROOT_DIR/data/geospatial/SSL4EO \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.5 \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name LESSVIT_b${EMBED_DIMS}_d${DECODER_DEPTH}_r${RANK} \
    --lr_scheduler cosine \
    --channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_depth $DECODER_DEPTH \
    --decoder_out_chans 15 \
    --use_perception_field_mask \
    --max_grad_norm 1.0 \
    --proj_drop 0.1 \
    --attn_drop 0.1 \
    --drop_path_rate 0.1 \
    --loss_type mse \
    --modal_mode multi \
    --scale 1 \
    --crop_size 128 \
    --init_values 1.0 \
    --rank $RANK
    
