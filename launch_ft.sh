for i in 1e-3
do
    for j in 1e-4
    do  
        echo "ViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mae_base \
        --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        echo "CViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mae_vit_btnk \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
