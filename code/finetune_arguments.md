# Finetuning(shoulder) : densenet 121
```
set SAVE_DIR=results/shoulder_mae/densenet121/randomcrop/models/
set lOG_DIR=results/shoulder_mae/densenet121/randomcrop/
set DATASET_DIR=data/DB_X-ray/

python ./code/finetune_shoulderxray.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--data_path %DATASET_DIR% ^
--model densenet121 ^
--checkpoint_type "smp_encoder" ^
--drop_path 0 
--finetune "best_models/densenet121_SHDR_1.4K_mae_random_800epc.pth"
```

# Finetuning(shoulder) - ViT-s

set SAVE_DIR=results/shoulder_mae/vitsmall/random_boundingbox/models/
set lOG_DIR=results/shoulder_mae/vitsmall/random_boundingbox/
set DATASET_DIR=data/DB_X-ray_rotated/

python ./code/finetune_shoulderxray.py ^
--output_dir %SAVE_DIR% ^
--log_dir %LOG_DIR% ^
--data_path %DATASET_DIR% ^
--model vit_small_patch16 ^
--layer_decay 0.55 ^
--drop_path 0.2 ^
--finetune "best_models/vis-s_SHDR_1.4K_mae_random_bbox_800epc.pth"
```

--mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--build_timm_transform ^
--num_workers 4 ^
--warmup_epochs 5 ^
--min_lr 1e-5 ^
--weight_decay 0.05 ^
--batch_size 8 ^
--accum_iter 4 ^
--epochs 80 ^
--blr 2.5e-4 
--aa rand-m6-mstd0.5-inc1 ^
