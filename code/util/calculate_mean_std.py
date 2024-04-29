import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

def Calculate_mean_std(dataset):
    """ 데이터셋의 평균과 표준편차를 계산
    그레이스케일 이미지에 대해 계산하고 3채널로 확장"""
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0

    print("Calculating means and stds for the dataset...")
    for images, _ in dataloader:
        mean += images[:,0,:,:].mean()
        std += images[:,0,:,:].std()

    mean /= len(dataloader)
    std /= len(dataloader)

    # 3채널로 확장
    mean = torch.full((3,), mean)
    std = torch.full((3,), std)

    return mean, std

