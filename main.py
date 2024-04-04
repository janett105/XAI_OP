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

from torchvision.models import resnet50, ResNet50_Weights

def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--model', default=resnet50, help='model')
    parser.add_argument('--weights', default=ResNet50_Weights.DEFAULT, help='pretrained model weights')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()

class Solver(object):
    def __init__(self, config):
        self.device = None
        self.weights = config.weights
        self.model = config.model
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.lr = config.lr
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.cuda = config.cuda
        self.data_loaders = None
        self.data_sets = None
        self.data_transforms = None
        self.n_folds=config.n_folds
        self.skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        self.metrics_df = pd.DataFrame(columns=['acc'])
    
    def load_data(self):
        self.data_transforms = {
            "train" : transforms.Compose([self.weights.transforms(),
                                          transforms.RandomHorizontalFlip()]),
            'val' : transforms.Compose([self.weights.transforms()]),
            'test' : transforms.Compose([self.weights.transforms()])
        }
        self.data_sets={
            #datasets.ImageFolder(root=os.path.join(DATA_DIR, datasets_type), transform=data_transforms[datasets_type]) 
            'train_val':torchvision.datasets.CIFAR10(root='./data/train_val', train=True, transform=None, download=True),
            'test':torchvision.datasets.CIFAR10(root='./data/test', train=False, transform=self.data_transforms['test'], download=True)
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
    
    def split_train_val(self, train_idx, val_idx):
        train_subset = Subset(self.data_sets['train_val'], train_idx)
        train_subset.dataset = copy.deepcopy(self.data_sets['train_val'])
        train_subset.dataset.transform = self.data_transforms['train']

        val_subset = Subset(self.data_sets['train_val'], val_idx)
        val_subset.dataset = copy.deepcopy(self.data_sets['train_val'])
        val_subset.dataset.transform = self.data_transforms['val']

        self.data_loaders['train'] = DataLoader(train_subset, batch_size=self.train_batch_size,shuffle=True,pin_memory=True,drop_last=False, num_workers=4)
        self.data_loaders['val'] = DataLoader(val_subset, batch_size=self.train_batch_size,shuffle=False, pin_memory=True,drop_last=False, num_workers=4)
    
    def train(self):
        print("training..")
        self.model.train()
        train_loss=0.0
        train_correct = 0
        total=0
        for _, (data_batch, labels) in enumerate(self.train_loader):
            data_batch, labels = data_batch.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() # parameter gradients 0으로 초기화
            outputs = self.model(data_batch)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            prediction = torch.max(outputs, 1)  # second param "1" represents the dimension to be reduced
            total += labels.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())# train_correct incremented by one if predicted right
        return train_loss, train_correct / total

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
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data_batch)
                
                valtest_loss += self.criterion(outputs, labels).item()
                prediction = torch.max(outputs, 1)
                total += labels.size(0)
                valtest_correct += np.sum(prediction[1].cpu().numpy() == labels.cpu().numpy())

        return valtest_loss, valtest_correct / total
    
    def save_model_state(self, fold):
        path = f"models/resnet/{fold}Fold.pth"
        torch.save(self.model.state_dict(), path)
        print("Checkpoint saved to {}".format(path))

    def test(self, fold):
        best_model=self.model().to(self.device)
        best_model.load_state_dict(torch.load(f'models/resnet/{fold}Fold.pth'))
        return evaluate(best_model, mode='test')

    def run(self):
        self.load_data()
        self.load_model()
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(range(len(self.data_sets['train_val'])), self.data_sets['train_val'].targets)):
            print(f"======={fold+1} fold=======")
            self.split_train_val(train_idx, val_idx)
            best_val_acc = 0.0
            for epoch in range(1, self.epochs+1):
                self.scheduler.step(epoch)
                print(f"\n===> epoch:{epoch}/{self.epochs}")
                train_loss, train_acc = train()
                print(f"train result : {train_loss, train_acc}")
                val_loss, val_acc = evaluate(mode='val')
                print(f"val result : {val_loss, val_acc}")
                if best_val_acc < val_acc:
                    self.save_model_state(fold)
                    best_val_acc = val_acc
            print(f"===> {fold}Fold BEST VAL_ACC: {best_val_acc * 100:.3f}")
            test_loss, test_acc = self.test(fold)
            print(f"===> {fold}Fold TEST_ACC: {test_acc * 100:.3f}")
            self.metrics_df.loc[fold, 'acc'] = test_acc

        print(f"10 Fold TEST_Acc:\n{self.metrics_df}")
        print(f'Final test Acc: {self.metrics_df["acc"].mean():.3f} ({self.metrics_df["acc"].std():.3f})')

if __name__ == '__main__':
    main()