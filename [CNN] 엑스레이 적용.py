# 1epoch씩 (train, val)반복
# 전체 epoch동안 가장 best인 val_acc일 때 model, acc저장
# 모든 epoch 끝난 후 최고일 때의 model로 test
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torchvision
from torchvision import transforms
from tutorial.train_evaluate_test import train, evaluate
from torch.utils.data import DataLoader, Subset
import copy
import argparse
import numpy as np
import sys
import visdom #python -m visdom.server
import os
#GRADCAM 라이브러리###############
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import cv2
import os

#from torchvision.models import resnet50, ResNet50_Weights # pretrain :  ImageNet
"""
mae pretraining : based on chest X-ray MAE study
    masking 비율 90%
    random resize crop : 0.5~1.0, center/random masking
    center/random masking

proxy pretraining : 


fine tuning : based on chest X-ray MAE study
    lr : 1.5e-4/1.5e-5/1.5e-6
    lr schedular : cosine annealing strategy
    optimizer : AdamW (beta1=0.9, beta2=0.95)
    layer-wise LR decay : 0.55
    RandAug magnitude : 6
    DropPath rate : 0.2
    n_epoch : 75 
"""
def main():

    parser = argparse.ArgumentParser(description="CNN/ViT X-ray classification")
    parser.add_argument('--lr', default=1.5e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1, type=int, help='number of epochs to train for')#에폭수 변경
    parser.add_argument('--trainBatchSize', default=16, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=16, type=int, help='testing batch size')
    parser.add_argument('--n_folds', default=10, type=int, help='the number of folds in stratified k fold')
    parser.add_argument('--model', default='cnn', type=str, help='model')
    parser.add_argument('--cuda', action='store_false', help='disable cuda')  # CUDA 사용 안 함 옵션 추가
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()
    
    # GradCAM 설정###############
    model = solver.model  # Solver 클래스에서 학습된 모델을 가져옵니다.
    target_layer = model.features.norm5
    cam = GradCAM(model=model, target_layers=[target_layer])

    # CAM 이미지 생성 및 저장
    output_folder = "results/cam_images"  # CAM 이미지를 저장할 폴더
    data_loader = solver.data_loaders['test']  # 테스트 데이터 로더를 사용
    generate_and_save_cam_images(data_loader, model, cam, output_folder)

    sys.stdout.close()

class Solver(object):
    def __init__(self, config):
        self.device = torch.device('cpu')  # 기본적으로 CPU 사용
        self.weights = config.weights
        self.model = config.model
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.lr = config.lr
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.cuda = config.cuda  # CUDA 설정 사용
        self.data_loaders = None
        self.data_sets = None
        self.data_transforms = None
        self.n_folds=config.n_folds
        self.skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        self.metrics_df = pd.DataFrame(columns=['acc'])
    
    def load_data(self):
        self.data_transforms = {
            "train" : transforms.Compose([transforms.RandAugment(magnitude=6)]),
            'val' : transforms.Compose([]),
            'test' : transforms.Compose([])
        }
        self.data_sets={
            'train':torchvision.datasets.ImageFolder(root='data/DB_X-ray/train_to', transform=self.data_transforms['train']),
            'val':torchvision.datasets.ImageFolder(root='data/DB_X-ray/val_to', transform=self.data_transforms['val']),
            'test':torchvision.datasets.ImageFolder(root='data/DB_X-ray/test_to', transform=self.data_transforms['test']) 
            # 'train_val':torchvision.datasets.CIFAR10(root='./data/train_val', train=True, transform=None, download=True),
            # 'test':torchvision.datasets.CIFAR10(root='./data/test', train=False, transform=self.data_transforms['test'], download=True)
        }
        self.data_loaders = {
                            'train': DataLoader(self.data_sets['train'], batch_size=self.train_batch_size, shuffle=True,num_workers=4),
                            'val': DataLoader(self.data_sets['val'], batch_size=self.test_batch_size, shuffle=False,num_workers=4),
                            'test': DataLoader(self.data_sets['test'], batch_size=self.test_batch_size, shuffle=False,num_workers=4)
                            }

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else: self.device = torch.device('cpu')
        print(f"device : {self.device}")
        
        if self.model=='cnn':
            self.model = torch.load('models/densenet121_CXR_0.3M_mae.pth').to(self.device)
            print('densenet151 : MAE(chest X-ray)')
        elif self.model=='vit':
            self.model = torch.load('models/vit-s_CXR_0.3M_mae.pth').to(self.device)
            print('ViT-small : MAE(chest X-ray)')
        elif self.model=='cnn_proxy':

            print('densenet151 : MAE(chest X-ray) -> rot_proxy(shoulder X-ray)')
        elif self.model=='vit_proxy':

            print('ViT-small : MAE(chest X-ray) -> rot_proxy(shoulder X-ray)')
        elif self.model=='cnn_rnd_mae':

            print('densenet151 : MAE(chest X-ray) -> MAE(shoulder X-ray, random masking/crop)')
        elif self.model=='vit_rnd_mae':

            print('ViT-small : MAE(chest X-ray) -> MAE(shoulder X-ray, random masking/crop)')
        elif self.model=='cnn_cnt_mae':

            print('densenet151 : MAE(chest X-ray) -> MAE(shoulder X-ray, center masking/crop)')
        elif self.model=='vit_cnt_mae':

            print('ViT-small : MAE(chest X-ray) -> MAE(shoulder X-ray, center masking/crop)')
        
        print(f'Learning Rate : {self.lr}')
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer)
    
    def split_train_val(self, train_idx, val_idx):
        train_subset = Subset(self.data_sets['train_val'], train_idx)
        train_subset.dataset = copy.deepcopy(self.data_sets['train_val'])
        train_subset.dataset.transform = self.data_transforms['train']

        val_subset = Subset(self.data_sets['train_val'], val_idx)
        val_subset.dataset = copy.deepcopy(self.data_sets['train_val'])
        val_subset.dataset.transform = self.data_transforms['val']

        # pin memory: DRAM을 거치지 않고 GPU 전용 메모리（VRAM）에 데이터를 로드
        # drop_last: batch의 크기에 따른 의존도 높은 함수를 사용할 때 걱정이 되는 경우 마지막 batch를 사용하지 않기
        # num_workers: 데이터 로딩에 사용하는 subprocess개수（멀티프로세싱）
        self.data_loaders['train'] = DataLoader(train_subset, batch_size=self.train_batch_size,shuffle=True,pin_memory=True,drop_last=False, num_workers=4)
        self.data_loaders['val'] = DataLoader(val_subset, batch_size=self.train_batch_size,shuffle=False, pin_memory=True,drop_last=False, num_workers=4)
    
    def train(self):
        print("training..")
        self.model.train()
        train_loss=0.0
        train_correct = 0
        total=0
        for _, (data_batch, labels) in enumerate(self.data_loaders['train']):
            data_batch, labels = data_batch.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() # parameter gradients 0으로 초기화
            outputs = self.model(data_batch)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            # 모델의 예측값 중 가장 높은 값을 가지는 인덱스를 구함
            prediction = torch.max(outputs, 1)  # second param "1" represents the dimension to be reduced
            total += labels.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())# train_correct incremented by one if predicted right
        return train_loss / len(self.data_loaders['train']), train_correct / total

    def evaluate(self, mode):
        print("testing..")
        self.model.eval()
        valtest_loss = 0.0
        valtest_correct = 0
        total = 0
        if mode=='val':valtest_loader = self.data_loaders['val']
        else:valtest_loader = self.data_loaders['test']
        with torch.no_grad():
            for _, (data_batch, labels) in enumerate(valtest_loader):
                data_batch, labels = data_batch.to(self.device), labels.to(self.device)
                outputs = self.model(data_batch)
                
                valtest_loss += self.criterion(outputs, labels).item()
                prediction = torch.max(outputs, 1)
                total += labels.size(0)
                valtest_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())
                
        # len(valtest_loader)로 나눔
        return valtest_loss / len(valtest_loader), valtest_correct / total
    
    def save_model_state(self, fold):
        directory = f"results/resnet"
        if not os.path.exists(directory):
            os.makedirs(directory)  # 디렉토리가 없다면 생성
        path = f"{directory}/{fold}Fold.pth"
        torch.save(self.model.state_dict(), path)
        print("Checkpoint saved to {}".format(path))

    def test(self, fold):
        #self.model은 이미 모델의 인스턴스니까 () 필요 없음        
        self.model.load_state_dict(torch.load(f'results/resnet/{fold}Fold.pth'))
        test_loss, test_acc = self.evaluate(mode='test')  # 모델을 평가
        return test_loss, test_acc

    def run(self):
        print(f'<finetuning for {self.n_folds} folds & {self.epochs} epochs>')

        self.load_data()
        self.load_model()
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(range(len(self.data_sets['train_val'])), self.data_sets['train_val'].targets)):
            print(f"======={fold+1} fold=======")
            self.split_train_val(train_idx, val_idx)
            best_val_acc = 0.0
            for epoch in range(1, self.epochs+1):
                print(f"\n===> epoch:{epoch}/{self.epochs}")
                train_loss, train_acc = self.train()
                print(f"train result : {train_loss, train_acc}")
                val_loss, val_acc = self.evaluate(mode='val')
                print(f"val result : {val_loss, val_acc}")
                if best_val_acc < val_acc:
                    self.save_model_state(fold)
                    best_val_acc = val_acc
                self.scheduler.step() # PyTorch의 더 깨끗하고 직관적인 API 사용을 장려하기 위해 torch.optim.lr_scheduler의 사용법이 변경되어 epoch 삭제
            print(f"===> {fold}Fold BEST VAL_ACC: {best_val_acc * 100:.3f}")
            test_loss, test_acc = self.test(fold)
            print(f"===> {fold}Fold TEST_ACC: {test_acc * 100:.3f}")
            self.metrics_df.loc[fold, 'acc'] = test_acc

        print(f"10 Fold TEST_Acc:\n{self.metrics_df}")
        print(f'Final test Acc: {self.metrics_df["acc"].mean():.3f} ({self.metrics_df["acc"].std():.3f})')
      
####GRADCAM 함수#####################################################################        
# 폴더 생성 함수
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# CAM 이미지 생성 및 저장 함수
def generate_and_save_cam_images(data_loader, model, cam, output_folder):
    create_directory(output_folder)  # 폴더 생성
    model.eval()
    for i, (images, _) in enumerate(data_loader):
        rgb_img = np.float32(images.squeeze().numpy().transpose((1, 2, 0)))
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cv2.imwrite(f'{output_folder}/cam_image_{i}.jpg', cam_image * 255)

if __name__ == '__main__':
    main()