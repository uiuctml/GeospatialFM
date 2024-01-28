# for i in 5e-4
# do
#     for j in 5e-2
#     do  
#         echo "learning_rate: $i, weight_decay: $j"
#         CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mae_base \
#         --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
#         MODEL.load_pretrained_from=dir \
#         TRAINER.learning_rate=$i TRAINER.weight_decay=$j
#     done
# done

for i in 5e-4
do
    for j in 5e-2
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10080 -m finetune --exp_name mae_cvit \
        --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
