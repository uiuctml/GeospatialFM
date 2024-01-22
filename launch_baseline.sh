for i in 1e-4 1e-3
do
    for j in 5e-2
    do  
        echo "learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10085 -m finetune_baseline --exp_name mae_baseline_lr$i --config_file GeospatialFM/configs/finetune_vit.yaml --debug TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
