for i in 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1
do
    for j in 0
    do  
        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_vit_v2 \
        # --config_file GeospatialFM/configs/lp_vit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit_0-2-10 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2 

        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit_2-0-10 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0

        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit_2-1-9 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

        # CUDA_VISIBLE_DEVICES=0 python -m lp --exp_name mae_cvit_1-2-9 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

        CUDA_VISIBLE_DEVICES=4 python -m lp --exp_name mm_lr_vit_1-2-9 \
        --config_file GeospatialFM/configs/lp_mm_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true MODEL.MULTI_MODAL.kwargs.dim_ratio=0.25

        # CUDA_VISIBLE_DEVICES=5 python -m lp --exp_name mm_lr_vit_1-10-1_fast \
        # --config_file GeospatialFM/configs/lp_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=10

        # CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_cvit_0-12-0 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=12

        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit_2-2-8 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit_0-12-0 \
        # --config_file GeospatialFM/configs/lp_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=12

        # CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_mm_cvit_1-2-9 \
        # --config_file GeospatialFM/configs/lp_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2
    done
done
