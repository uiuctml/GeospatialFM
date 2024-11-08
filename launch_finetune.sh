ROOT_DIR="/home/haozhesi/Dropbox/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

for lr in 3e-4; do
    for adam_wd in 0.01; do
        for checkpoint in 60000; do
            for num_experts in 1 3 5 10; do
                echo "lr: $lr, adam_wd: $adam_wd"
                accelerate launch GeospatialFM/finetune/finetune.py \
                --data_dir $ROOT_DIR/data/geospatial \
                --dataset_name eurosat \
                --task_type classification \
                --scale 2 \
                --modal optical \
                --return_dict \
                --per_device_train_batch_size 64 \
                --gradient_accumulation_steps 4 \
                --num_train_epochs 20 \
                --learning_rate $lr \
                --adam_weight_decay $adam_wd \
                --warmup_steps 0 \
                --warmup_ratio 0.2 \
                --report_to wandb \
                --save_total_limit 5 \
                --seed 42 \
                --mixed_precision bf16 \
                --dataloader_num_workers 32 \
                --dataloader_pin_memory \
                --output_dir $ROOT_DIR/results/models \
                --logging_dir $ROOT_DIR/results/logs \
                --wandb_dir $ROOT_DIR/results/ \
                --run_name LESSVIT_b2_d6_eurosat_lr${lr}_wd${adam_wd}_ckpt${checkpoint}_exp${num_experts} \
                --lr_scheduler_type cosine \
                --channel_embed_dims_per_head 2 \
                --use_perception_field_mask \
                --pretrained_model_path $ROOT_DIR/results/models/LESSVIT_b2_d6/checkpoint-${checkpoint}/model.safetensors \
                --use_moe \
                --num_experts $num_experts 
            done
        done
    done
done