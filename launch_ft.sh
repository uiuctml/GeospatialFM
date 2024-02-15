for i in 1e-3
do
    for j in 0
    do  
        # echo "ViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10088 -m finetune --exp_name mae_vit_v2 \
        # --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        echo "CViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_v2 \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
