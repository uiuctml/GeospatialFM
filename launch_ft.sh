CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=10086 -m finetune_mae --exp_name mae_base_dist --config_file GeospatialFM/configs/finetune_mae.yaml --debug
