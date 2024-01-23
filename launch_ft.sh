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

for i in 1e-4 5e-4
do
    for j in 5e-2
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10087 -m finetune --exp_name mae_base_si_maxpool \
        --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
