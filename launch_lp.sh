for i in 1e-1 5e-2 1e-2 5e-3 1e-3
do
    for j in 0
    do  
        # CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_vit \
        # --config_file GeospatialFM/configs/lp_vit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        CUDA_VISIBLE_DEVICES=0 python -m lp --exp_name mae_rca_vit \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done
