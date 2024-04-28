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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from tutorial.train_evaluate_test import train, evaluate
from torch.utils.data import DataLoader, Subset
from PIL import Image
import pandas as pd
import copy
import argparse
import numpy as np
import sys
import visdom #python -m visdom.server
import os

from torchvision.models import vit_b_16, ViT_B_16_Weights # pretrain : IMAGENET1K_V1

DATA_DIR = './data/Shoulder_X-rayDB/DB_X-ray'


def main():
    #sys.stdout = open('logs.pth', 'w')

    parser = argparse.ArgumentParser(description="shoulder x-ray")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=32, type=int, help='training batch size') # 16, 32, 64
    parser.add_argument('--testBatchSize', default=32, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--n_folds', default=10, type=int, help='the number of folds in stratified k fold')
    parser.add_argument('--model', default=vit_b_16, help='model')
    parser.add_argument('--weights', default=ViT_B_16_Weights.DEFAULT, help='pretrained model weights')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()

    #sys.stdout.close()
    
class Solver(object):
    def __init__(self, config):
        self.device = None
        self.weights = config.weights
        self.model = config.model
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.lr = config.lr
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.cuda = config.cuda
        self.data_loaders = None
        self.data_sets = None
        self.data_transforms = None
        self.n_folds=config.n_folds
        self.skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        self.metrics_df = pd.DataFrame(columns=['acc'])
        
        # python -m visdom.server -port 8097
        # http://localhost:8097
        self.vis = visdom.Visdom(port=8097)
        self.loss_win = None
        self.acc_win = None
    
    def load_data(self):
        self.data_transforms = {
            'train' : transforms.Compose([self.weights.transforms(),
                                          transforms.RandomHorizontalFlip(),]),
            'val' : transforms.Compose([self.weights.transforms(),]),
            'test' : transforms.Compose([self.weights.transforms(),])
        }
        
        self.data_sets={
            'train_val': datasets.ImageFolder(
                root=os.path.join(DATA_DIR, 'train_val')
                ),
            'test': datasets.ImageFolder(
                root=os.path.join(DATA_DIR, 'test')
                , transform=self.data_transforms['test']) 
        }
        self.data_loaders = {'train': [],
                'val': [],
                'test': DataLoader(self.data_sets['test'], batch_size=self.test_batch_size, shuffle=False,num_workers=4)
        }

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else: self.device = torch.device('cpu')
        print(f"device : {self.device}")

        self.model = self.model(weights=self.weights).to(self.device)
        #model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
    
    def split_train_val(self, train_idx, val_idx):
        train_subset = Subset(self.data_sets['train_val'], train_idx)
        train_subset.dataset = copy.deepcopy(self.data_sets['train_val'])
        train_subset.dataset.transform = self.data_transforms['train']

        val_subset = Subset(self.data_sets['train_val'], val_idx)
        val_subset.dataset = copy.deepcopy(self.data_sets['train_val'])
        val_subset.dataset.transform = self.data_transforms['val']

        # pin_memory: DRAM을 거치지 않고 GPU 전용 메모리（VRAM）에 데이터를 로드
        # drop_last: batch의 크기에 따른 의존도 높은 함수를 사용할 때 걱정이 되는 경우 마지막 batch를 사용하지 않기
        # num_workers: 데이터 로딩에 사용하는 subprocess개수（멀티프로세싱）
        self.data_loaders['train'] = DataLoader(train_subset, batch_size=self.train_batch_size,shuffle=True,pin_memory=True,drop_last=False, num_workers=4)
        self.data_loaders['val'] = DataLoader(val_subset, batch_size=self.train_batch_size,shuffle=False, pin_memory=True,drop_last=False, num_workers=4)
     
    def train(self,fold):
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
            train_correct += (prediction[1] == labels).sum().item()# train_correct incremented by one if predicted right
            
        train_loss = train_loss / len(self.data_loaders['train'])
        train_acc = train_correct / total   
        
        # Loss와 Accuracy를 Visdom에 기록
        # 에포크 번호와 훈련 손실을 기반으로 점을 그립니다.
        # X 배열은 [에포크 번호, 손실 값] 형태의 2열을 가져야 합니다.
        x_val = np.array([self.current_epoch])
        y_val = np.array([train_loss])
        X = np.column_stack((x_val, y_val))
        
        if self.loss_win is None:
            self.loss_win = self.vis.scatter(X=X,
                                             opts=dict(title='Train Loss',
                                                       xlabel='Epoch',
                                                       ylabel='Loss',
                                                       legend=['Train Loss']))
        else:
            self.vis.scatter(X=X,
                     win=self.loss_win,
                     update='append')
            
        x_val = np.array([self.current_epoch])
        y_val = np.array([train_acc])
        Y = np.column_stack((x_val, y_val))
        if self.acc_win is None:
            self.acc_win = self.vis.scatter(X=Y,
                                             opts=dict(title='Train Accuracy',
                                                       xlabel='Epoch',
                                                       ylabel='Accuracy',
                                                       legend=['Train Accuracy']))
        else:
            self.vis.scatter(X=Y,
                     win=self.acc_win,
                     update='append')
            
        
            
        return train_loss / len(self.data_loaders['train']), train_correct / total

    def evaluate(self, mode):
        if(mode=="test"):
            print("testing..")
        else: 
            print("training..")
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
                # valtest_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())
                valtest_correct += (prediction[1] == labels).sum().item()
                
        # len(valtest_loader)로 나눔
        return valtest_loss / len(valtest_loader), valtest_correct / total
    
    def save_model_state(self, fold):
        directory = f"results/vit_b_16"
        if not os.path.exists(directory):
            os.makedirs(directory)  # 디렉토리가 없다면 생성
        path = f"{directory}/{fold+1}Fold.pth"
        torch.save(self.model.state_dict(), path)
        print("Checkpoint saved to {}".format(path))

    def test(self, fold):
        #self.model은 이미 모델의 인스턴스니까 () 필요 없음        
        self.model.load_state_dict(torch.load(f'results/vit_b_16/{fold+1}Fold.pth'))
        test_loss, test_acc = self.evaluate(mode='test')  # 모델을 평가
        return test_loss, test_acc

    def run(self):
        self.load_data()
        self.load_model()
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(range(len(self.data_sets['train_val'])), self.data_sets['train_val'].targets)):
            print(f"======={fold+1} fold=======")
            self.split_train_val(train_idx, val_idx)
            best_val_acc = 0.0
            for epoch in range(1, self.epochs+1):
                self.current_epoch = epoch  # 현재 에포크 저장
                print(f"\n===> epoch:{epoch}/{self.epochs}")
                train_loss, train_acc = self.train(fold)
                print(f"train result : {train_loss, train_acc}")
                val_loss, val_acc = self.evaluate(mode='val')
                print(f"val result : {val_loss, val_acc}")
                if best_val_acc < val_acc:
                    self.save_model_state(fold)
                    best_val_acc = val_acc
                self.scheduler.step() # PyTorch의 더 깨끗하고 직관적인 API 사용을 장려하기 위해 torch.optim.lr_scheduler의 사용법이 변경되어 epoch 삭제
            print(f"===> {fold+1}Fold BEST VAL_ACC: {best_val_acc * 100:.3f}")
            test_loss, test_acc = self.test(fold)
            print(f"===> {fold+1}Fold TEST_ACC: {test_acc * 100:.3f}")
            self.metrics_df.loc[fold, 'acc'] = test_acc

        print(f"10 Fold TEST_Acc:\n{self.metrics_df}")
        print(f'Final test Acc: {self.metrics_df["acc"].mean():.3f} ({self.metrics_df["acc"].std():.3f})')

if __name__ == '__main__':
    main()