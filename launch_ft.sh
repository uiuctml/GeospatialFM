for i in 1e-3
do
    for j in 1e-2
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node 2 --master_port=10086 -m finetune_mae --exp_name mae_base_dist --config_file GeospatialFM/configs/finetune_mae.yaml --debug TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
