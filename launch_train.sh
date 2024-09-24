export PYTHONPATH=$PYTHONPATH:/home/haozhesi/Dropbox/GeospatialFM
NCCL_BLOCKING_WAIT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch GeospatialFM/scripts/train.py \
    --data_dir /home/haozhesi/Dropbox/GeospatialFM/data/geospatial/SSL4EO \
    --train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
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
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --output_dir /home/haozhesi/Dropbox/GeospatialFM/results/models \
    --logging_dir /home/haozhesi/Dropbox/GeospatialFM/results/logs \
    --wandb_dir /home/haozhesi/Dropbox/GeospatialFM/results/ \
    --run_name LRSSVIT-ViTDecoder-clstoken2 \
    --lr_scheduler cosine \
    --use_perception_field_mask \
    --channel_embed_dims_per_head 1 \
    --decoder_channel_embed_dims_per_head 1 \
    --decoder_depth 12 \
    --decoder_out_chans 15 \
    # --modal_mode None \
    # --norm_pix_loss \
    # --use_8bit \
    # --resume_from_checkpoint /home/haozhesi/Dropbox/GeospatialFM/pretrained_models/mae_pretrained_model.pth \
