# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m train_dist.py --exp_name mae_base_siglip1e-2 --config_file GeospatialFM/configs/mae_cm.yaml
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port=10085 -m train_dist --exp_name mae_base_warmup --config_file GeospatialFM/configs/mae_cm.yaml 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=10085 -m train_dist --exp_name mae_base_warmup_spec --config_file GeospatialFM/configs/cvit_mae.yaml

