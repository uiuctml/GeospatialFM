for i in 5e-3 1e-2 2e-2 4e-2 5e-2:
do
    CUDA_VISIBLE_DEVICES=2 python train.py --config_file GeospatialFM/configs/eurosat.yaml TRAINER.learning_rate=$i TRAINER.per_device_train_batch_size=256
done