CUDA_VISIBLE_DEVICES=2 python train.py --exp_name dino_frac0.1 --config_file GeospatialFM/configs/eurosat/eurosat_vit-s16_dino.yaml TRAINER.learning_rate=0.1 MODEL.freeze_encoder=true
# CUDA_VISIBLE_DEVICES=2 python train.py --exp_name moco_frac0.1 --config_file GeospatialFM/configs/eurosat/eurosat_vit-s16_moco.yaml TRAINER.learning_rate=0.1 
# CUDA_VISIBLE_DEVICES=3 python train.py --exp_name seco_frac0.1 --config_file GeospatialFM/configs/eurosat/eurosat_rn50_seco.yaml TRAINER.learning_rate=0.1 
# CUDA_VISIBLE_DEVICES=2 python train.py --exp_name moco_frac0.1 --config_file GeospatialFM/configs/eurosat/eurosat_rn50_moco.yaml TRAINER.learning_rate=0.1 
# CUDA_VISIBLE_DEVICES=3 python train.py --exp_name dino_frac0.1 --config_file GeospatialFM/configs/eurosat/eurosat_rn50_dino.yaml TRAINER.learning_rate=0.1 

