for i in 7e-5 6e-5 5e-5
do
    for j in 6e-3 5e-3 4e-3
    do  
        #echo "ViT learning_rate: $i, weight_decay: $j"
        #CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mae_vit_btnk \
        #--config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        #MODEL.load_pretrained_from=dir \
        #TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        echo "CViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10084 -m finetune --exp_name mae_vit_btnk_attn_v2 \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
