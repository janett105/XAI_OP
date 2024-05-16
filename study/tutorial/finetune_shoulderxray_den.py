import datetime
import json
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
#from util.mixup_multi_label import Mixup

from util.multi_label_loss import SoftTargetBinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_shoulder_xray
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit

from engine_finetune import train_one_epoch,evaluate_shoulderxray
from util.sampler import RASampler
#from apex.optimizers import FusedAdam
from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict

def finetune_cnn(args):
    #misc.init_distributed_mode(args)

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    # if args.device:
    #     device = torch.device('cuda')
    #     cudnn.benchmark = True
    # else: device = torch.device('cpu')
    # print(f"device : {device}")

    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset_shoulder_xray(split='train', args=args)
    dataset_val = build_dataset_shoulder_xray(split='val', args=args)
    dataset_test = build_dataset_shoulder_xray(split='test', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # if args.log_dir is not None and not args.eval:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup is activated!")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if 'densenet' in args.model:
        model = models.__dict__[args.model](num_classes=args.nb_classes)
    else:
        raise NotImplementedError
    
    # binary classification을 위한 model classifier(FC layer) 변경
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 1, bias=True)
    
    if args.finetune and not args.eval:
        if 'densenet' in args.model:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
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
            print(msg)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # elif args.optimizer == 'fusedlamb':
    #     optimizer = FusedAdam(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.dataset == 'shoulderxray':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # if
    # criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if args.eval:
        test_stats = evaluate_shoulderxray(data_loader_test, model, device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        print(f"Accuracy of the network on the test set images: {test_stats['acc1']:.4f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auc_accuracy = 0.0
    max_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.eval_interval == 0 or epoch + 1 == args.epochs):
            val_stats = evaluate_shoulderxray(data_loader_val, model, device, args)
            print(f"Average AUC on the val set images: {val_stats['auc_avg']:.4f}")
            print(f"Accuracy of the network on val images: {val_stats['acc1']:.1f}%")

            if max_auc< val_stats['auc_avg']:
                max_auc = val_stats['auc_avg']
                max_auc_accuracy = val_stats['acc1']
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
    test_stats = evaluate_shoulderxray(data_loader_test, model, device, args)
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