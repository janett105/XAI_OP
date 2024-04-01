from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import os
import copy

from torchvision.datasets import ImageFolder, CIFAR10 # datasets size = 50,000(train) 10,000(test)
DATA_DIR = 'data/'
IMAGE_DIM = 227 # pixels
BATCH_SIZE = 100 # 5000개 batch

data_transforms = {
    "train" : transforms.Compose([
        # validation에서도 가능한 transformation만..?
        transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
    'val' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
    'test' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}

data_sets={'train_val':CIFAR10(root=os.path.join(DATA_DIR, 'train_val'), train=True, transform=None, download=False),
          'test':CIFAR10(root=os.path.join(DATA_DIR, 'test'), train=False, transform=data_transforms['test'], download=False)}
# image_datasets = {datasets_type: datasets.ImageFolder(root=os.path.join(DATA_DIR, datasets_type), transform=data_transforms[datasets_type]) 
#                   for datasets_type in['train', 'val']}
print('\nDataset created')

data_loaders = {'train': [],
                'val': [],
                'test': DataLoader(data_sets['test'], batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=4)}# num_workers: CPU가 GPU에 data 올릴 때 사용할 subprocess 개수(0이면 main process 하나만 사용), batch size가 클 땐 num_workers를 줄여야 함

def split_train_val(train_idx, val_idx):
    DataLoader(data_sets['train_val'], batch_size=BATCH_SIZE,shuffle=True, 
                                    num_workers=8, pin_memory=True,drop_last=False),
    train_subset = Subset(data_sets['train_val'], train_idx)
    train_subset.dataset = copy.deepcopy(data_sets['train_val'])
    train_subset.dataset.transform = data_transforms['train']

    val_subset = Subset(data_sets['train_val'], val_idx)
    val_subset.dataset = copy.deepcopy(data_sets['train_val'])
    val_subset.dataset.transform = data_transforms['val']

    data_loaders['train'] = DataLoader(train_subset, batch_size=BATCH_SIZE,shuffle=True,
                                    pin_memory=True,drop_last=False, num_workers=4)
    data_loaders['val'] = DataLoader(val_subset, batch_size=BATCH_SIZE,shuffle=False, 
                                    pin_memory=True,drop_last=False, num_workers=4)
    
    print('Dataloader created')