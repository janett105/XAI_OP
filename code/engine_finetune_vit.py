import math
import sys
from typing import Iterable, Optional
 
import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy
import torch.distributed as dist
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics._ranking import roc_auc_score
import torch.nn.functional as F
from libauc import losses
from sklearn.metrics import confusion_matrix
import copy
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, last_activation=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # 타겟을 one-hot 벡터로 변환
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=2).to(torch.float32).to(device)

        # print('samples', samples.shape, 'targets', targets.shape)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if last_activation is not None:
            if last_activation == 'sigmoid':
                last_activation = torch.nn.Sigmoid()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # print('outputs', outputs.shape, 'targets', targets.shape)
            # print(outputs.shape, targets.shape, torch.unique(targets))
            if last_activation is not None:
                outputs = last_activation(outputs)
            loss = criterion(outputs, targets_one_hot)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # 타겟을 one-hot 벡터로 변환
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=2).to(torch.float32).to(device)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target_one_hot)

        pred = (output.sigmoid() > 0.5).float()
        correct = (pred == target).float().sum()
        accuracy = correct / images.size(0)
        
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item(), accuracy=accuracy.item())
        
        #metric_logger.update(loss=loss.item())
        #metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Accuracy {accuracy.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(accuracy=metric_logger.accuracy, losses=metric_logger.loss))
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #     .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def computeAUROC(dataGT, dataPRED, classCount):
    if classCount==2:
        try:
            outAUROC = roc_auc_score(dataGT, dataPRED)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            print(f"dataGT: {dataGT}, dataPRED: {dataPRED}")
            outAUROC = 0.  # or np.nan to indicate failure

    return outAUROC


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

@torch.no_grad()
def evaluate_shoulderxray(data_loader, model, device, args):
    if args.dataset == 'shoulderxray':
        criterion = torch.nn.BCEWithLogitsLoss()
        #criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

       # 타겟을 one-hot 벡터로 변환
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=2).to(torch.float32).to(device)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target_one_hot)

        acc = accuracy(output, target, topk=(1, ))[0]
        outputs.append(output)
        targets.append(target_one_hot)
        

        #accuracy = 100 * correct / len(target)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc.item(), n=batch_size)
        # metric_logger.update(loss=loss.item(), accuracy=accuracy.item())

    # gather the stats from all processes
    # print(len(outputs), outputs[0].shape)
    # outputs = collect_results_gpu(outputs, len(data_loader.dataset))
    # targets = collect_results_gpu(targets, len(data_loader.dataset))
    # print(outputs_collected)
    # rank = dist.get_rank()
    # if rank == 0:
    #     outputs_gathered = [output.to(device) for output in outputs]
    #     targets_gathered = [target.to(device) for target in targets]
    #     outputs = torch.cat(outputs_gathered, dim=0).sigmoid().cpu().numpy()
    #     targets = torch.cat(targets_gathered, dim=0).cpu().numpy()
    #     auc_each_class = computeAUROC(targets, outputs, 14)
    #     auc_avg = np.average(auc_each_class)

    num_classes = args.nb_classes
    outputs = torch.cat(outputs, dim=0).sigmoid().cpu().numpy() # batch별 결과 합침
    targets = torch.cat(targets, dim=0).cpu().numpy()
    print(f'total targets shape : {targets.shape}\ntotal outputs shape : {outputs.shape}')
    
    np.save(args.log_dir + '/' + 'y_gt.npy', targets)
    np.save(args.log_dir + '/' + 'y_pred.npy', outputs)
   
    auc_each_class = computeAUROC(targets, outputs, num_classes)
    auc_each_class_array = np.array(auc_each_class)
    missing_classes_index = np.where(auc_each_class_array == 0)[0]
    # print(missing_classes_index)
    if missing_classes_index.shape[0] > 0:
        print('There are classes that not be predicted during testing,'
              ' the indexes are:', missing_classes_index)

    auc_avg = np.average(auc_each_class_array[auc_each_class_array != 0])
    metric_logger.synchronize_between_processes()

    print('Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))
    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()},
            **{'auc_avg': auc_avg, 'auc_each_class': auc_each_class}}


    mapping = {
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print(matrix)

    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]

    print('Sens', ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
    print('PPV', ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))