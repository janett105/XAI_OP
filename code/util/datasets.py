# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.dataloader_med import RetinaDataset, Augmentation, Node21, ChestX_ray14, Covidx, CheXpert
from .custom_transforms import GaussianBlur
import torch
from .augment import new_data_aug_generator
from util.imagerotation import ImageRotation

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_shoulder_xray(split, args):
    is_train = (split == 'train')
    is_train_rotation = (split == 'train_rotation')
    # transform = build_transform(is_train, args)
    if args.build_timm_transform:
        transform = build_transform(is_train, args)
    else:
        if is_train:
            if args.aug_strategy == 'imagerotation':
                transform = build_transform(True, args)
            elif args.aug_strategy == 'simclr_with_randrotation':
                print(args.aug_strategy)
                transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomRotation(degrees=(0, 45)),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
                ])
            
            elif args.aug_strategy == 'threeaugment':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
                transform = new_data_aug_generator(args, mean=mean, std=std)
            elif args.aug_strategy == 'default':
                transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "train")
            else:
                raise NotImplementedError
        elif is_train_rotation:
            if args.aug_strategy == 'imagerotation':
                print(args.aug_strategy)
                transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                ImageRotation(degrees=[0, 90, 180, 270]),
                transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
                ])
        else:
            transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")

    if args.dataset == 'shoulderxray':
        if split == 'train':mode = 'train'
        elif split == 'val':mode = 'validation'
        else:mode= 'test'
        
        dataset = datasets.ImageFolder(root=f'data/DB_X-ray/{mode}_to', transform=transform)
        
        print("dataset:: ", dataset)
        return dataset

def build_transform(is_train, args):
    if args.norm_stats is not None:
        if args.norm_stats == 'imagenet':
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            raise NotImplementedError
    else:
        try:
            if args.dataset == 'chestxray' or args.dataset == 'covidx' or args.dataset == 'chexpert':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
            elif args.dataset == 'imagenet':
                mean = IMAGENET_DEFAULT_MEAN
                std = IMAGENET_DEFAULT_STD
            elif args.dataset == 'retina':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
            elif args.dataset == 'shoulderxray':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)               
        except:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD

    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)