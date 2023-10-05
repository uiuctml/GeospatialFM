for i in 5e-2 6e-2 7e-2:
do
    CUDA_VISIBLE_DEVICES=2 python train.py --config_file GeospatialFM/configs/eurosat.yaml TRAINER.learning_rate=$i TRAINER.per_device_train_batch_size=256
done