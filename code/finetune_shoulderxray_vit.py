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

from util.multi_label_loss import SoftTargetBinaryCrossEntropy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_shoulder_xray
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models import models_vit

from engine_finetune_vit import train_one_epoch,evaluate_shoulderxray
from util.sampler import RASampler
#from apex.optimizers import FusedAdam
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict
import sys

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=75, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.55,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    # 1e-5, 1e-6, 1e-7 시도하기
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m6-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='models/vit-s_CXR_0.3M_mae.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='data/DB_X-ray/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='results/shoulder_mae/vit/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='results/shoulder_mae/vit/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', action='store_false', 
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_false', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
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
    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument("--build_timm_transform", action='store_true', default=True)
    parser.add_argument("--aug_strategy", default='simclr_with_randrotation', type=str, help="strategy for data augmentation")
    parser.add_argument("--dataset", default='shoulderxray', type=str)

    parser.add_argument('--repeated_aug', action='store_true', default=False)

    parser.add_argument("--optimizer", default='adamw', type=str)

    parser.add_argument('--ThreeAugment', action='store_false')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument("--norm_stats", default=None, type=str)

    parser.add_argument("--checkpoint_type", default=None, type=str)

    parser.add_argument("--distributed", default=False, type=bool)

    return parser

def finetune_vit(args):
    # misc.init_distributed_mode(args)
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir) # tenrsorboard : 시각화
        writer = open(file=args.log_dir+'training_logs.txt', mode='w') # log, print(출력물, file=writer)
    else:log_writer = None

    if args.device == torch.device('cuda'):
        cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
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
        drop_last=True,)

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
    # mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup is activated!")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    if 'vit' in args.model:
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_rate=args.vit_dropout_rate,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune, file=writer)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in checkpoint_model.keys():
            if k in state_dict:
                if checkpoint_model[k].shape == state_dict[k].shape:
                    state_dict[k] = checkpoint_model[k]
                    print(f"Loaded Index: {k} from Saved Weights")
                else:
                    print(f"Shape of {k} doesn't match with {state_dict[k]}")
            else:
                print(f"{k} not found in Init Model")

        # interpolate position embedding
        # model의 input size/구조가 달라졌을 때 필요
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        # strict=False : 완벽한 일치 아니어도 됨
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(args.device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#------------------------------- Learning Parameters 준비---------------------------------------------------
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # if mixup_fn is not None:
    #     criterion = SoftTargetBinaryCrossEntropy()
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else: 
        criterion = torch.nn.BCEWithLogitsLoss()

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print("criterion = %s" % str(criterion), file=writer)
    print("Model = %s" % str(model_without_ddp), file=writer)
    print('number of params (M): %.2f' % (n_parameters / 1.e6), file=writer)
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size), file=writer)
    print("actual lr: %.2e" % args.lr, file=writer)
    print("accumulate grad iterations: %d" % args.accum_iter, file=writer)
    print("effective batch size: %d" % eff_batch_size, file=writer)

#------------------------------- Training & Evaluation---------------------------------------------------
    if args.eval:
        test_stats = evaluate_shoulderxray(data_loader_test, model, args.device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        print(f"Accuracy of the network on the test set images: {test_stats['acc1']:.4f}")
        exit(0)

    # print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # max_accuracy = 0.0
    # max_auc = 0.0
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         data_loader_train.sampler.set_epoch(epoch)
    #     train_stats = train_one_epoch(
    #         model, criterion, data_loader_train,
    #         optimizer, args.device, epoch, loss_scaler,
    #         args.clip_grad, log_writer=log_writer, args=args)

    #     if args.output_dir and (epoch % args.eval_interval == 0 or epoch + 1 == args.epochs):
    #         val_stats = evaluate_shoulderxray(data_loader_val, model, args.device, args)
    #         print(f"Average AUC on the val set images: {val_stats['auc_avg']:.4f}")
    #         print(f"Accuracy of the network on val images: {val_stats['acc1']:.1f}%")
            
    #         if val_stats['auc_avg'] > max_auc or val_stats['acc1'] > max_accuracy:
    #             max_auc = max(max_auc,val_stats['auc_avg'])
    #             max_accuracy = max(max_accuracy,val_stats['acc1'])
                # misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #                 loss_scaler=loss_scaler, epoch=epoch)
            
    #         if log_writer is not None:
    #             log_writer.add_scalar('perf/auc_avg', val_stats['auc_avg'], epoch)
    #             log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            
    #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                         **{f'val_{k}': v for k, v in val_stats.items()},
    #                         'epoch': epoch,
    #                         'n_parameters': n_parameters}

    #         if args.output_dir and misc.is_main_process():
    #             if log_writer is not None:
    #                 log_writer.flush()
    #             with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #                 f.write(json.dumps(log_stats) + "\n")



    checkpoint = torch.load(os.path.join(args.output_dir,'bestval_model.pth'), map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in checkpoint_model.keys():
        if k in state_dict:
            if checkpoint_model[k].shape == state_dict[k].shape:
                state_dict[k] = checkpoint_model[k]
                print(f"Loaded Index: {k} from Saved Weights")
            else:
                print(f"Shape of {k} doesn't match with {state_dict[k]}")
        else:
            print(f"{k} not found in Init Model")
    model.load_state_dict(checkpoint_model, strict=False)

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
#     path = f"{args.output_dir}/bestval_model.pth"
#     torch.save(model.state_dict(), path)
#     print("Checkpoint saved to {}".format(path))

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    finetune_vit(args)