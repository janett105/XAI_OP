# Pretraining(shoulder)/MAE : densenet 121
```
set SAVE_DIR=results/shoulder_mae/densenet121/centercrop

python ./code/pretrain_shoulderxray_cnn.py ^
 --output_dir %SAVE_DIR% ^
 --log_dir %SAVE_DIR% ^
 --mask_ratio 0.75 ^
 --epochs 800 ^
 --warmup_epochs 40 ^
 --blr 1.5e-4 --weight_decay 0.05 ^
 --num_workers 2 ^
 --input_size 224 ^
 --random_resize_range 0.5 1.0 ^
 --datasets_names shoulder_xray ^
 --checkpoint_type "smp_encoder" ^
 --distributed ^
 --crop_ratio 0.8
 --device
```
# Pre-training(shoulder)/MAE - ViT-S
```
python -m torch.distributed.launch --nproc_per_node=8 ^
 --use_env main_pretrain_multi_datasets_xray.py ^
 --output_dir ${SAVE_DIR} ^
 --log_dir ${SAVE_DIR} ^
 --model 'mae_vit_small_patch16_dec128d2b' ^
 --mask_ratio 0.90 ^
 --epochs 800 ^
 --warmup_epochs 40 ^
 --blr 1.5e-4 --weight_decay 0.05 ^
 --num_workers 8 ^
 --input_size 224 ^
 --mask_strategy 'random' ^
 --random_resize_range 0.5 1.0 ^
 --datasets_names chexpert chestxray_nih
```

