# CUDA_VISIBLE_DEVICES=1 python train.py --exp_name dino_frac0.1 --config_file GeospatialFM/configs/bigearthnet/bn_vit-s16_dino.yaml &
# CUDA_VISIBLE_DEVICES=2 python train.py --exp_name moco_frac0.1 --config_file GeospatialFM/configs/bigearthnet/bn_vit-s16_moco.yaml &
# CUDA_VISIBLE_DEVICES=3 python train.py --exp_name dino_frac0.1 --config_file GeospatialFM/configs/bigearthnet/bn_rn50_dino.yaml &
# CUDA_VISIBLE_DEVICES=4 python train.py --exp_name moco_frac0.1 --config_file GeospatialFM/configs/bigearthnet/bn_rn50_moco.yaml &
CUDA_VISIBLE_DEVICES=3 python train.py --exp_name seco_frac0.1 --config_file GeospatialFM/configs/bigearthnet/bn_rn50_seco.yaml

