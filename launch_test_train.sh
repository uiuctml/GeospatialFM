ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch GeospatialFM/scripts/train.py \
    --data_dir $ROOT_DIR/data/geospatial/SSL4EO \
    --train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 20 \
    --learning_rate 1.5e-4 \
    --adam_weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.5 \
    --lr_warmup_steps 2000 \
    --report_to wandb \
    --save_steps 500 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory \
    --output_dir $ROOT_DIR/results/models \
    --logging_dir $ROOT_DIR/results/logs \
    --wandb_dir $ROOT_DIR/results/ \
    --run_name LRSSVIT-2-2-2 \
    --lr_scheduler cosine \
    --channel_embed_dims_per_head 2 \
    --decoder_channel_embed_dims_per_head 2 \
    --decoder_depth 2 \
    --decoder_out_chans 15 \
    --use_perception_field_mask \
    --early_stop_steps 8000 \
    --resume_from_checkpoint latest \
    # --modal_mode multi
    # --max_grad_norm 1.0 \
    # --norm_pix_loss \
    # --use_8bit \
    # --resume_from_checkpoint /home/haozhesi/Dropbox/GeospatialFM/pretrained_models/mae_pretrained_model.pth \
