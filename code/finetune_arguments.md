
# Finetuning(shoulder) : densenet 121
```
set SAVE_DIR=results/shoulder_mae/densenet121/centercrop/
set DATASET_DIR=data/DB_X-ray/

python ./code/finetune_shoulderxray_cnn.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--finetune "models/densenet121_SHDR_center_1.4K_mae_800epc.pth"
--checkpoint_type "smp_encoder" ^
--blr 2.5e-4 ^
--model "densenet121" ^
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--data_path %DATASET_DIR% ^
--min_lr 1e-5 ^
--repeated_aug ^
```
# Finetuning(shoulder) - ViT-s

```
set SAVE_DIR=results/shoulder_mae/vit/
set DATASET_DIR=data/DB_X-ray/

python ./code/finetune_shoulderxray_vit.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--finetune "models/vit-s_CXR_0.3M_mae.pth"
--checkpoint_type "smp_encoder" ^
--blr 2.5e-4 ^
--model "vit_small_patch16" ^
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--data_path %DATASET_DIR% ^
--min_lr 1e-5 ^
--repeated_aug ^
```
