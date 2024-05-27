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
set SAVE_DIR=results/shoulder_mae/vitsmall/centercrop_heatmap

python ./code/pretrain_shoulderxray_vit.py ^
 --output_dir %SAVE_DIR% ^
 --log_dir %SAVE_DIR% ^
 --mask_ratio 0.90 ^
 --epochs 800 ^
 --warmup_epochs 40 ^
 --blr 1.5e-4 --weight_decay 0.05 ^
 --num_workers 2 ^
 --input_size 224 ^
 --random_resize_range 0.5 1.0 ^
 --datasets_names shoulder_xray ^
 --batch_size 8 ^
 --accum_iter 4 ^
 --mae_strategy 'heatmap_mask_boundingbox'
```
tensorboard --logdir=%SAVE_DIR%