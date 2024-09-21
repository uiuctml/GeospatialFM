export PYTHONPATH=$PYTHONPATH:/home/haozhesi/Dropbox/GeospatialFM
NCCL_BLOCKING_WAIT=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch GeospatialFM/scripts/train.py \
    --data_dir /home/haozhesi/Dropbox/GeospatialFM/data/geospatial/SSL4EO \
    --train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --adam_weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.5 \
    --lr_warmup_steps 1000 \
    --report_to wandb \
    --save_steps 1000 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --output_dir /home/haozhesi/Dropbox/GeospatialFM/results/models \
    --logging_dir /home/haozhesi/Dropbox/GeospatialFM/results/logs \
    --wandb_dir /home/haozhesi/Dropbox/GeospatialFM/results/ \
    --run_name LRSSVIT \
    --use_perception_field_mask \
    --use_8bit \
    --norm_pix_loss \
    # --resume_from_checkpoint /home/haozhesi/Dropbox/GeospatialFM/pretrained_models/mae_pretrained_model.pth \


