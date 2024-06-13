# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

from util import misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_mae_vit as models_mae_vit
from models.models_mae_cnn import MaskedAutoencoderCNN
#from models.models_mae_cnn import MaskedAutoencoderCNN

from engine_pretrain import train_one_epoch
from util.dataloader_med import ShoulderXray
# import cv2
from util.custom_transforms import custom_train_transform
from util.sampler import RASampler

from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch16_dec512d8b', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.90, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='data/DB_X-ray_rotated/train_to', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='results/shoulder_mae/vitsmall/centercrop_heatmap/models/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='results/shoulder_mae/vitsmall/centercrop_heatmap/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', action='store_false', 
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--train_list", default=None, type=str, help="file for training list")
    parser.add_argument('--random_resize_range', type=float, nargs='+', default=[0.5, 1.0],
                        help='RandomResizedCrop min/max ratio, default: None)')
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--repeated-aug', action='store_true', default=False)
    parser.add_argument('--datasets_names', type=str, nargs='+', default=['shoulder_xray'])
    parser.add_argument('--distributed', default=False, action='store_true')
    
    parser.add_argument('--mask_strategy', default='random', type=str)
    parser.add_argument('--finetune', default='best_models/vit-s_CXR_0.3M_mae.pth',
                        help='finetune from checkpoint')
    parser.add_argument("--mae_strategy", default='heatmap_mask_boundingbox', type=str)
    parser.add_argument("--checkpoint_type", default=None, type=str)
    return parser

def main(args):
    if 'vit' in args.model:
        assert timm.__version__ == "0.3.2"  # version check

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.device == torch.device('cuda'):
        cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    print(f"device : {args.device}")

    # simple augmentation

    # if args.resize_input == -1:
    #     transform_train = transforms.Compose([
    #             transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])])
    #
    # else:
    #     scaled_ratio_min = 0.2 * args.resize_input / 1024
    #     scaled_ratio_max = 1.0 * args.resize_input / 1024
    concat_datasets = []
    mean_dict = {'chexpert': [0.485, 0.456, 0.406],
                 'chestxray_nih': [0.5056, 0.5056, 0.5056],
                 'mimic_cxr': [0.485, 0.456, 0.406],
                 'shoulder_xray': [0.5056, 0.5056, 0.5056]
                 }
    std_dict = {'chexpert': [0.229, 0.224, 0.225],
                'chestxray_nih': [0.252, 0.252, 0.252],
                'mimic_cxr': [0.229, 0.224, 0.225],
                'shoulder_xray': [0.252, 0.252, 0.252]
                }
    print(args.datasets_names)

    for dataset_name in args.datasets_names:
        dataset_mean = mean_dict[dataset_name]
        dataset_std = std_dict[dataset_name]
        print(args.mae_strategy)
        
        if args.mae_strategy=='random_crop_mask' and args.random_resize_range: # randomcrop -> resize -> randommasking
            resize_ratio_min, resize_ratio_max = args.random_resize_range
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(resize_ratio_min, resize_ratio_max),
                                                interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)])
        elif args.mae_strategy in ['random_mask_boundingbox', 'heatmap_mask_boundingbox']:
            # resize-> random masking in bounding box
            # resize -> heat map masking based on bounding box
            args.mask_strategy = 'heatmap_weighted'

            if args.mae_strategy =='heatmap_mask_boundingbox': heatmap_path = 'data/DB_BBox/Bbox.npy'
            elif args.mae_strategy =='random_mask_boundingbox':
                if 'vit' in args.model : heatmap_path = 'data/DB_BBox/BboxLine_2.2.npy'
                elif 'densenet' in args.model : heatmap_path = 'data/DB_BBox/BboxLine_1.9.npy'

            transform_train = custom_train_transform(size=args.input_size, mean=dataset_mean, std=dataset_std)

        # if args.mask_strategy in ['heatmap_weighted', 'heatmap_inverse_weighted']:
        #     heatmap_path = 'data/DB_BBox/mask_heatmap.png' #nih_bbox_heatmap.png
        #     heatmap = cv2.imread(heatmap_path)
        # else:
        #     heatmap_path = None

        # if dataset_name == 'chexpert':
        #     dataset = CheXpert(csv_path="data/CheXpert-v1.0-small/train.csv", image_root_path='data/CheXpert-v1.0-small/', use_upsampling=False,
        #                        use_frontal=True, mode='train', class_index=-1, transform=transform_train,
        #                        heatmap_path=heatmap_path, pretraining=True)
        # elif dataset_name == 'chestxray_nih':
            # dataset = ChestX_ray14('data/nih_chestxray', "data_splits/chestxray/train_official.txt", augment=transform_train, num_class=14,
            #                        heatmap_path=heatmap_path, pretraining=True)
        # elif dataset_name == 'mimic_cxr':
        #     dataset = MIMIC(path='data/mimic_cxr', version="chexpert", split="train", transform=transform_train, views=["AP", "PA"],
        #                     unique_patients=False, pretraining=True)
        # else:
        #     raise NotImplementedError
        
        # concat_datasets.append(dataset)
    #dataset_train = torch.utils.data.ConcatDataset(concat_datasets)
    dataset_train = ShoulderXray(args.data_path, transform=transform_train, heatmap_path=heatmap_path)
    print(dataset_train)

    # if args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     if args.repeated_aug:
    #         sampler_train = RASampler(
    #             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #         )
    #     else:
    #         sampler_train = torch.utils.data.DistributedSampler(
    #             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #         )
        
    global_rank=0
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

# ------------------------MODEL-----------------------------------------------------
    if 'vit' in args.model :
        model = models_mae_vit.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=args.input_size, 
                                                    mask_strategy=args.mask_strategy)
        if args.finetune:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in checkpoint_model.keys():
                if k in state_dict:
                    if checkpoint_model[k].shape == state_dict[k].shape:
                        state_dict[k] = checkpoint_model[k]
                        print(f"Loaded Index: {k} from Saved Weights")
                    else:
                        print(f'{k} 문제 \n 참조 모델 : {checkpoint_model[k].shape} \n 현재 모델 :{state_dict[k].shape}')
                        #print(f"Shape of {k} doesn't match with {state_dict[k]}")
                else:
                    print(f"{k} not found in Init Model")

            # interpolate position embedding
            # model의 input size/구조가 달라졌을 때 필요
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model, decoder는 load X
            # strict=False : 완벽한 일치 아니어도 됨
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # if args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            # encoder의 마지막 layer는 바꿀 필요 X
            #trunc_normal_(model.head.weight, std=2e-5)
    elif 'densenet' in args.model :
        model = MaskedAutoencoderCNN(checkpoint_type=args.checkpoint_type, img_size=args.input_size, patch_size=16, 
                                     model_arch='Unet', encoder_name='densenet121',
                                    pretrained_path='models/densenet121_CXR_0.3M_mae.pth',
                                    mask_strategy=args.mask_strategy)
     
    model.to(args.device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    try: 
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    except: # timm 버전 호환
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)\
        
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, args.device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, pretrain=True)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "score.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)