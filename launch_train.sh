# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10085 -m train_dist --exp_name mae_base_warmup_spec_maxpool_unichannel --config_file GeospatialFM/configs/cvit_mae.yaml
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m train_dist --exp_name mae_base_si_maxpool --config_file GeospatialFM/configs/cvit_mae.yaml
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_0-0-12 \
#     --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
#     MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=0 

# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10083 -m train --exp_name mae_cvit_2-2-8 \
    # --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    # MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10087 -m train --exp_name mae_cvit_2-0-10 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.kwargs.spectral_blocks=2 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=10

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_0-2-10 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.kwargs.spectral_blocks=0 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=2

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10086 -m train --exp_name mae_cvit_1-1-10 \
#     --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
#     MODEL.OPTICAL.kwargs.spectral_blocks=1 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=1

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_4-4-4 \
#     --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
#     MODEL.OPTICAL.kwargs.spectral_blocks=4 MODEL.OPTICAL.kwargs.sptial_spectral_blocks=4