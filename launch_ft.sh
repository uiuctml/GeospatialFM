for i in 5e-4 5e-3 5e-5
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
        # CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_1-1-10 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

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

        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_mm_cvit_1-2-9 \
        --config_file GeospatialFM/configs/finetune_mm_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2
    done
done
