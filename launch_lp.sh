for i in 1e-2
do
    for j in 0
    do  
        CUDA_VISIBLE_DEVICES=1 python -m lp --exp_name mae_cvit \
        --config_file GeospatialFM/configs/lp_vit.yaml --debug \
        MODEL.load_pretrained_from=dir MODEL.pretrained_ckpt=ckpt_epoch10.pth \
        TRAINER.learning_rate=$i TRAINER.weight_decay=$j
    done
done


# for i in 1e-3
# do
#     for j in 0
#     do  
#         CUDA_VISIBLE_DEVICES=3 python -m lp --exp_name mae_baseline \
#         --config_file GeospatialFM/configs/lp_vit.yaml --debug \
#         MODEL.load_pretrained_from=timm MODEL.pretrained_ckpt=mae \
#         TRAINER.learning_rate=$i TRAINER.weight_decay=$j
#     done
# done
