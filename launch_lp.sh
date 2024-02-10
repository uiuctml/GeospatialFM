for i in 9e-2
do
    for j in 0
    do  
        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_vit_v2 \
        --config_file GeospatialFM/configs/lp_vit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_v2 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
