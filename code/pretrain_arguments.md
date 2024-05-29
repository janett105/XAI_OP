# Pretraining(shoulder)/MAE : densenet 121
```
set SAVE_DIR=results/shoulder_mae/vitsmall/centercrop_heatmap/models/
set lOG_DIR=results/shoulder_mae/vitsmall/centercrop_heatmap/
set DATASET_DIR=data/DB_X-ray_rotated/

python ./code/pretrain_shoulderxray_cnn.py ^
 --output_dir %SAVE_DIR% ^
 --log_dir %LOG_DIR% ^
 --data_path %DATASET_DIR% ^
 --model 'densenet121' ^
 --mask_ratio 0.75 ^
 --checkpoint_type "smp_encoder" ^
 --mae_strategy heatmap_mask_boundingbox
```
# Pre-training(shoulder)/MAE - ViT-S
```
set SAVE_DIR=results/shoulder_mae/vitsmall/centercrop_heatmap/models/
set lOG_DIR=results/shoulder_mae/vitsmall/centercrop_heatmap/
set DATASET_DIR=data/DB_X-ray_rotated/

python ./code/pretrain_shoulderxray_vit.py ^
 --output_dir %SAVE_DIR% ^
 --log_dir %LOG_DIR% ^
 --data_path %DATASET_DIR% ^
 --mask_ratio 0.90 ^
 --model 'mae_vit_small_patch16_dec512d8b' ^
 --finetune 'best_models/vit-s_CXR_0.3M_mae.pth' ^
 --mae_strategy heatmap_mask_boundingbox
```

tensorboard --logdir=%SAVE_DIR%

 --warmup_epochs 40 ^
 --blr 1.5e-4 --weight_decay 0.05 ^
 --epochs 800 ^
 --num_workers 2 ^
 --input_size 224 ^
 --datasets_names shoulder_xray ^
 --batch_size 8 ^
 --accum_iter 4 ^