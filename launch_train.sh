# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10085 -m train_dist --exp_name mae_base_warmup_spec_maxpool_unichannel --config_file GeospatialFM/configs/cvit_mae.yaml
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m train_dist --exp_name mae_base_si_maxpool --config_file GeospatialFM/configs/cvit_mae.yaml
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10082 -m train --exp_name mae_base_vit --config_file GeospatialFM/configs/pretrain_vit.yaml
