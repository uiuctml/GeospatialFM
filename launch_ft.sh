# for i in 1e-3
# do
#     for j in 5e-2
#     do  
#         echo "learning_rate: $i, weight_decay: $j"
#         CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mae_base \
#         --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
#         MODEL.load_pretrained_from=dir \
#         TRAINER.learning_rate=$i TRAINER.weight_decay=$j
#     done
# done

for i in 3e-5
do
    for j in 6e-3
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10087 -m finetune --exp_name mae_base \
        --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        echo "CViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mae_rca_vit \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
