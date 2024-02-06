for i in 5e-3 1e-3 5e-4
do
    for j in 5e-2 5e-3
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_baseline \
        --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        MODEL.load_pretrained_from=timm MODEL.pretrained_ckpt=mae \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
