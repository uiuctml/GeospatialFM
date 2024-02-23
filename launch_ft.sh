for i in 5e-4 5e-5 
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

        echo "CViT learning_rate: $i, weight_decay: $j"
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_0-0-12 \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0 

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_4-4-4 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=4 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=4 \
        # TRAINER.per_device_train_batch_size=64 TRAINER.gradient_accumulation_steps=16

        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_1-1-10 \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_1-0-11 \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0

        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_0-1-11 \
        --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        MODEL.load_pretrained_from=dir \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

        # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10089 -m finetune --exp_name mae_cvit_0-12-0 \
        # --config_file GeospatialFM/configs/finetune_cvit.yaml --debug \
        # MODEL.load_pretrained_from=dir \
        # TRAINER.learning_rate=$i TRAINER.weight_decay=$j \
        # MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=12 \
        # TRAINER.per_device_train_batch_size=32 TRAINER.gradient_accumulation_steps=32
    done
done
