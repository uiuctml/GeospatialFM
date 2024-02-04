for i in 1e-2 5e-2
do
    for j in 5e-2
    do  
        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_base \
        --config_file GeospatialFM/configs/lp_vit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_vit_btnk \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
