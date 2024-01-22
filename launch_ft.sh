for i in 1e-3
do
    for j in 5e-2
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m finetune_mae --exp_name mae_base \
        --config_file GeospatialFM/configs/finetune_mae.yaml --debug \
        MODEL.load_pretrain_from=timm MODEL.pretrain_ckpt=mae \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
