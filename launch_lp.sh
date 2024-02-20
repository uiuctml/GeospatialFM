for i in 5e-2
do
    for j in 0
    do  
        # CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_vit_v2 \
        # --config_file GeospatialFM/configs/lp_vit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit_0-0-12 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0 

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_4-4-4 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=4 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=4

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_1-1-10 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_1-0-11 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_0-1-11 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

        CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_0-12-0 \
        --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=12
    done
done
