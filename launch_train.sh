# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10085 -m train_dist --exp_name mae_base_warmup_spec_maxpool_unichannel --config_file GeospatialFM/configs/cvit_mae.yaml
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m train_dist --exp_name mae_base_si_maxpool --config_file GeospatialFM/configs/cvit_mae.yaml
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_0-0-12 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.spectral_blocks=0 MODEL.OPTICAL.sptial_spectral_blocks=0 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_0-12-0 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.spectral_blocks=0 MODEL.OPTICAL.sptial_spectral_blocks=12

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_0-1-11 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.spectral_blocks=0 MODEL.OPTICAL.sptial_spectral_blocks=1

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_1-0-11 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.spectral_blocks=1 MODEL.OPTICAL.sptial_spectral_blocks=0 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_1-1-10 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.spectral_blocks=1 MODEL.OPTICAL.sptial_spectral_blocks=1 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_cvit_4-4-4 \
    --config_file GeospatialFM/configs/pretrain_cvit_v2.yaml \
    MODEL.OPTICAL.spectral_blocks=4 MODEL.OPTICAL.sptial_spectral_blocks=4