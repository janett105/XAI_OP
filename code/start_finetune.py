from pathlib import Path
import argparse
import os
from study.tutorial.finetune_shoulderxray_den import finetune_cnn
from finetune.finetune_shoulderxray_vit import finetune_vit
import torch
import util.misc as misc

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default=f'densenet121', type=str, metavar='MODEL',
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
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
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
    parser.add_argument('--finetune', default='models/densenet121_CXR_0.3M_mae.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    # parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='data/DB_X-ray/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    # parser.add_argument('--dist_eval', action='store_true', default=False,
    #                     help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='stosre_false', dest='pin_mem')

    # distributed training parameters
    parser.add_argument('--world_size', default=misc.get_world_size(), type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')
    # parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    # parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    # parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument("--build_timm_transform", action='store_false', default=True)
    parser.add_argument("--aug_strategy", default='simclr_with_randrotation', type=str, 
                        help="strategy for data augmentation") # (timm_transform이 아닐 때 진행)
    parser.add_argument("--dataset", default='shoulderxray', type=str)
    parser.add_argument('--repeated_aug', action='store_true', default=True)
    parser.add_argument("--optimizer", default='adamw', type=str)
    # parser.add_argument('--ThreeAugment', action='store_false')  # 3augment
    # parser.add_argument('--src', action='store_true')  # simple random crop
    parser.add_argument('--loss_func', default=None, type=str)
    parser.add_argument("--norm_stats", default=None, type=str)
    parser.add_argument("--checkpoint_type", default=None, type=str)
    #parser.add_argument("--distributed", default=False, type=bool)

    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if 'densenet' in args.model:
        finetune_cnn(args)
    elif 'vit' in args.model:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        finetune_vit(args)