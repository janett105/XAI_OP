# Finetuning(shoulder) : densenet 121
```
set SAVE_DIR=results/shoulder_mae/densenet121/centercrop/
set DATASET_DIR=data/DB_X-ray/

python ./code/finetune_shoulderxray_cnn.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--data_path %DATASET_DIR% ^
--model 'densenet121' ^
--checkpoint_type "smp_encoder" ^
--blr 2.5e-4 ^
--model "densenet121" ^
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--min_lr 1e-5 ^
--warmup_epochs 5 ^
--epochs 75 ^
--blr 2.5e-4 --weight_decay 0.05 ^
--num_workers 4 ^
--build_timm_transform ^
--aa 'rand-m6-mstd0.5-inc1'
--repeated_aug ^
--finetune "models/densenet121_SHDR_center_1.4K_mae_800epc.pth"
```

# Finetuning(shoulder) - ViT-s

set SAVE_DIR=results/shoulder_mae/vitsmall/randomcrop/models
setlOG_DIR=results/shoulder_mae/vitsmall/randomcrop
set DATASET_DIR=data/DB_X-ray/

python ./code/finetune_shoulderxray_vit.py ^
--output_dir %SAVE_DIR% ^
--log_dir %LOG_DIR% ^
--data_path %DATASET_DIR% ^
--blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 ^
--model vit_small_patch16 ^
--drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--min_lr 1e-5 ^
--batch_size 8 ^
--accum_iter 4 ^
--warmup_epochs 5 ^ 
--epochs 80 ^
--num_workers 4 ^
--aa 'rand-m6-mstd0.5-inc1' ^
--build_timm_transform ^
--finetune "best_models/vis-s_SHDR_1.4K_mae_800epc.pth"

