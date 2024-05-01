for i in 7e-4
do
    for j in 5e-4
    do  
        # echo "ViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10088 -m finetune --exp_name mae_vit_v2 \
        # --config_file GeospatialFM/configs/finetune_vit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_v2 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j

        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mm_lr_vit_1-0-11 \
        # --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=0 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true

        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mm_lr_vit_1-2-9_ms \
        # --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true

        echo "CViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mm_vit_1-2-9_d4_new \
        --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug --finetune_modal radar \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true MODEL.MULTI_MODAL.kwargs.dim_ratio=0.25

        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mm_lr_vit_1-2-9_d4 \
        # --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true MODEL.MULTI_MODAL.kwargs.dim_ratio=0.25

        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10086 -m finetune --exp_name mm_lr_vit_1-2-9_d2_avg \
        # --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true MODEL.MULTI_MODAL.kwargs.dim_ratio=0.5



        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10085 -m finetune --exp_name mm_lr_vit_1-2-9_fast \
        # --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2

        # echo "CViT learning_rate: $i, weight_decay: $j"
        # CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_0-0-12 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_2-0-10 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0 \

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_2-2-8 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_1-2-9 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_2-1-9 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_0-12-0 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=12 \
        # TRAINER.per_device_train_batch_size=32 TRAINER.gradient_accumulation_steps=32

        # CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_mm_cvit_1-2-9 \
        # --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2
    done
done
