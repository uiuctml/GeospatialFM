ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export NCCL_BLOCKING_WAIT=1
accelerate launch GeospatialFM/scripts/train.py \
    --data_dir $ROOT_DIR/data/geospatial/SSL4EO \
    --train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1.5e-4 \
    --adam_weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.5 \
    --lr_warmup_steps 80000 \
    --report_to wandb \
    --save_steps 1000 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name LESSVIT_b2 \
    --lr_scheduler cosine \
    --channel_embed_dims_per_head 2 \
    --decoder_channel_embed_dims_per_head 2 \
    --decoder_depth 8 \
    --decoder_out_chans 15 \
    --use_perception_field_mask \
    --resume_from_checkpoint latest \
    --max_grad_norm 1.0 \
    --proj_drop 0.1 \
    --attn_drop 0.1 \
    --loss_type mse \
    