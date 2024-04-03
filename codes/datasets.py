from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import os
import copy

from torchvision.datasets import ImageFolder, CIFAR10 # datasets size = 50,000(train) 10,000(test)

from torchvision.models import resnet50, ResNet50_Weights

DATA_DIR = 'data/'
IMAGE_DIM = 224 # pixels
BATCH_SIZE = 100 # 5000개 batch
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()


data_transforms = {
    "train" : transforms.Compose([preprocess()]),
    'val' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'test' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}

data_sets={'train_val':CIFAR10(root=os.path.join(DATA_DIR, 'train_val'), train=True, transform=None, download=True),
          'test':CIFAR10(root=os.path.join(DATA_DIR, 'test'), train=False, transform=data_transforms['test'], download=True)}
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