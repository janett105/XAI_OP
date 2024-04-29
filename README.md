# 보고서 제출
1. 계획서 (기한: ~3/31)

2. 중간보고서 (기한: ~4/30)

KCC/KSC 학술대회 논문제출 양식은 추후 학회에서 양식 공지되면 송부 예정

3. 최종보고서 (20쪽 이상)(기한: ~6/15)

[보고서 제출] 상기 보고서들은 기한내에 아래 캡디 전담조교와 산중교수님께 제출해야 합니다.

보고서 전담조교: 문준혁 학생(wnsgur1564@khu.ac.kr)

보고서 담당교수: 오민석 산중교수님(?minseok.oh@khu.ac.kr)

# XAI_OP
GOAL : classify osteoporosis

pytorch 2.2.0
python 3.10
torchvision 0.15.2

### 참고 자료
Transfer Learning for Computer Vision Tutorial
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


set SAVE_DIR=results/densenet121/
set DATASET_DIR=data/DB_X-ray/
python ./code/finetune_shoulderxray.py ^
--output_dir %SAVE_DIR% ^
--log_dir %SAVE_DIR% ^
--batch_size 8 ^
--accum_iter 4 ^
--finetune "models/densenet121_CXR_0.3M_mae.pth" ^
--checkpoint_type "smp_encoder" ^
--epochs 1 ^
--input_size 1000 ^
--blr 2.5e-4 --weight_decay 0.05 ^
--model "densenet121" ^
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 ^
--data_path %DATASET_DIR% ^
--num_workers 2 ^
--nb_classes 2 ^
--min_lr 1e-5 ^
--build_timm_transform ^
--aa "rand-m6-mstd0.5-inc1" ^
--repeated_aug^

--warmup_epochs 5 ^
--eval_interval 10 ^
![Uploading image.png…]()

