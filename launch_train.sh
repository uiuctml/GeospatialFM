CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m train_dist.py --exp_name mae_base_siglip --config_file GeospatialFM/configs/mae_cm.yaml
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 -m train_dist.py --exp_name mae_base_dist --config_file GeospatialFM/configs/mae.yaml

