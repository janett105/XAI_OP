# Pretraining(shoulder) -> Finetuning(shoulder) : densenet 121
```
set SAVE_DIR=results/shoulder_mae/densenet121/centercrop

python ./code/pretrain_shoulderxray_cnn.py ^
 --output_dir %SAVE_DIR% ^
 --log_dir %SAVE_DIR% ^
 --batch_size 8 ^
 --accum_iter 4 ^
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
```
set SAVE_DIR=results/shoulder_mae/densenet121/centercrop/
set DATASET_DIR=data/DB_X-ray/

python ./code/finetune_shoulderxray_cnn.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--batch_size 8 ^
--accum_iter 4 ^
--checkpoint_type "smp_encoder" ^
--epochs 75 ^
--input_size 224 ^
--blr 2.5e-4 --weight_decay 0.05 ^
--model "densenet121" ^
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--data_path %DATASET_DIR% ^
--num_workers 2 ^
--nb_classes 2 ^
--min_lr 1e-5 ^
--build_timm_transform ^
--aa "rand-m6-mstd0.5-inc1" ^
--repeated_aug ^
--warmup_epochs 5 ^
--eval_interval 10 ^
--finetune "models/densenet121_SHDR_center_1.4K_mae_800epc.pth"
```
# Pre-training(shoulder) - ViT-S
```
python -m torch.distributed.launch --nproc_per_node=8 ^
 --use_env main_pretrain_multi_datasets_xray.py ^
 --output_dir ${SAVE_DIR} ^
 --log_dir ${SAVE_DIR} ^
 --batch_size 8 ^
 --accum_iter 4 ^
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

# Finetuning(shoulder) - densenet121
```
set SAVE_DIR=results/densenet121/
set DATASET_DIR=data/DB_X-ray/
python ./code/finetune_shoulderxray.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--batch_size 8 ^
--accum_iter 4 ^
--checkpoint_type "smp_encoder" ^
--epochs 75 ^
--input_size 224 ^
--blr 2.5e-4 --weight_decay 0.05 ^
--model "densenet121" ^
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--data_path %DATASET_DIR% ^
--num_workers 2 ^
--nb_classes 2 ^
--min_lr 1e-5 ^
--build_timm_transform ^
--aa "rand-m6-mstd0.5-inc1" ^
--repeated_aug ^
--warmup_epochs 5 ^
--eval_interval 10 ^
--finetune "models/densenet121_SHDR_1.4K_mae_800epc.pth" ^
--device
```

# Finetuning(shoulder) - ViT-s