# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10085 -m train_dist --exp_name mae_base_warmup_spec_maxpool_unichannel --config_file GeospatialFM/configs/cvit_mae.yaml
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m train_dist --exp_name mae_base_si_maxpool --config_file GeospatialFM/configs/cvit_mae.yaml
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_0-0-12 \
#     --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
#     MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0 

# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10083 -m train --exp_name mae_cvit_2-2-8 \
    # --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10087 -m train --exp_name mae_cvit_2-0-10 \
#     --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
#     MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=10

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=10086 -m train --exp_name mm_lr_vit_0-8-4 \
#     --config_file GeospatialFM/configs/pretrain_lr_vit_bn.yaml --debug \
    # MODEL.MULTI_MODAL.kwargs.spectral_blocks=0 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=4

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=10086 -m train --exp_name mm_lr_vit_0-12-0 \
#     --config_file GeospatialFM/configs/pretrain_lr_vit_bn.yaml \
#     MODEL.MULTI_MODAL.kwargs.spectral_blocks=0 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=12

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=10080 -m train --exp_name mm_lr_vit_1-2-9 \
    --config_file GeospatialFM/configs/pretrain_lr_vit_bn.yaml \
    MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 MODEL.MULTI_MODAL.kwargs.low_rank_feature=true MODEL.MULTI_MODAL.kwargs.dim_ratio=0.5

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=10087 -m train --exp_name mm_lr_vit_1-10-1_fast \
#     --config_file GeospatialFM/configs/pretrain_lr_vit_bn.yaml \
#     MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=10

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=10086 -m train --exp_name mae_cvit_1-2-9 \
#     --config_file GeospatialFM/configs/pretrain_mm_cvit_bn.yaml \
#     MODEL.MULTI_MODAL.kwargs.spectral_blocks=1 MODEL.MULTI_MODAL.kwargs.sptial_spectral_blocks=2 

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_4-4-4 \
#     --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
#     MODEL.OPTICAL.kwargs.spectral_blocks=4 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=4
