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

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
#from util.mixup_multi_label import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from util.multi_label_loss import SoftTargetBinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_shoulder_xray
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models.models_vit

from engine_finetune_cnn import train_one_epoch,evaluate_shoulderxray, evaluate_chestxray
from util.sampler import RASampler
#from apex.optimizers import FusedAdam
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict
from torchvision.models import densenet121, DenseNet121_Weights # pretrain : IMAGENET1K_V1
import sys 

def finetune_cnn(args):
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir) # tenrsorboard : 시각화
        writer = open(file=args.log_dir, mode='w') # log, print(출력물, file=writer)
    else:log_writer = None
    
    if args.device == torch.device('cuda'):
        cudnn.benchmark = True
    
    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print(f"device : {args.device}")
    print("{}".format(args).replace(', ', ',\n'), file=writer)

#------------------------------- Dataset 준비---------------------------------------------------
    dataset_train = build_dataset_shoulder_xray(split='train', args=args, logger=writer)
    dataset_val = build_dataset_shoulder_xray(split='val', args=args, logger=writer)
    dataset_test = build_dataset_shoulder_xray(split='test', args=args, logger=writer)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)

#------------------------------- Model 준비---------------------------------------------------
    if 'densenet' in args.model :
        model = models.__dict__[args.model](num_classes=args.nb_classes)
    else:
        raise NotImplementedError
    
    if args.finetune and not args.eval:
        print("Load pre-trained checkpoint from: %s" % args.finetune, file=writer)
        if args.finetune=='imagenet':
            model(DenseNet121_Weights).to(args.device)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            
            if 'state_dict' in checkpoint.keys():
                checkpoint_model = checkpoint['state_dict']
            elif 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            if args.checkpoint_type == 'smp_encoder':
                state_dict = checkpoint_model

                new_state_dict = OrderedDict()

                for key, value in state_dict.items():
                    if 'model.encoder.' in key:
                        new_key = key.replace('model.encoder.', '')
                        new_state_dict[new_key] = value
                checkpoint_model = new_state_dict
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg, file=writer)

    # binary classification을 위한 model classifier(FC layer) 변경
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 1, bias=True)

    model.to(args.device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#------------------------------- Learning Parameters 준비---------------------------------------------------
    eff_batch_size = args.batch_size * args.accum_iter * args.world_size
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    print("Model = %s" % str(model_without_ddp), file=writer)
    print('number of params (M): %.2f' % (n_parameters / 1.e6), file=writer)
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size), file=writer)
    print("actual lr: %.2e" % args.lr, file=writer)
    print("accumulate grad iterations: %d" % args.accum_iter, file=writer)
    print("effective batch size: %d" % eff_batch_size, file=writer)
    print("criterion = %s" % str(criterion), file=writer)

#------------------------------- Training & Evaluation---------------------------------------------------
    if args.eval: # finetuning없이 바로 test
        test_stats = evaluate_shoulderxray(data_loader_test, model, args.device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        print(f"Accuracy of the network on the test set images: {test_stats['acc1']:.4f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, args.device, epoch, loss_scaler,
            args.clip_grad,log_writer=log_writer,
            args=args)

        if args.output_dir and (epoch % args.eval_interval == 0 or epoch + 1 == args.epochs):
            val_stats = evaluate_shoulderxray(data_loader_val, model, args.device, args)
            print(f"Average AUC on the val set images: {val_stats['auc_avg']:.4f}")
            print(f"Accuracy of the network on val images: {val_stats['acc1']:.1f}%")

            if val_stats['auc_avg'] > max_auc:
                max_auc = max(max_auc,val_stats['auc_avg'])
                max_accuracy = val_stats['acc1']
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                
            if log_writer is not None:
                log_writer.add_scalar('perf/auc_avg', val_stats['auc_avg'], epoch)
                log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
    
    model.load_state_dict(torch.load(f'results/{args.model}/bestval_model.pth'))
    test_stats = evaluate_shoulderxray(data_loader_test, model, args.device, args)
    print(f"Average AUC on the test set images: {test_stats['auc_avg']:.4f}")
    print(f"Accuracy of the network on test images: {test_stats['acc1']:.1f}%")
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                    'n_parameters': n_parameters}
    
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    sys.stdout.close()

# def save_model_state(model):
#     directory = f"results/{args.model}"
#     if not os.path.exists(directory):
#         os.makedirs(directory)  # 디렉토리가 없다면 생성
#     path = f"{directory}/bestval_model.pth"
#     torch.save(model.state_dict(), path)
#     print("Checkpoint saved to {}".format(path))