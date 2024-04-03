# 1epoch씩 (train, val)반복
# 전체 epoch동안 가장 best인 val_acc일 때 model, acc저장
# 모든 epoch 끝난 후 최고일 때의 model로 test
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from codes.train_evaluate_test import train, evaluate
from codes.datasets import data_loaders, data_sets, split_train_val
# from models.AlexNet import AlexNet
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")

N_FOLDS = 2
DATA_DIR = 'data/'
MODEL_NAME = 'AlexNet'

N_EPOCHS = 10
MOMENTUM = 0.9 # 최소 0.9
LR_DECAY = 0.0001
LR_INIT = 0.01

BATCH_SIZE = 100 # 500개 batch
IMAGE_DIM = 224 # pixels
N_LABELS = 10
N_IN_CHANNELS=3 #RGB
N_CLASSES = 10
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

METRICS=['acc']


def save_best_model(model, fold):
    path = f"models/best_{MODEL_NAME}_{fold}Fold.pth"
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    metrics_df = pd.DataFrame(columns=METRICS)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_sets['train_val'])), data_sets['train_val'].targets)):
        print(f"======================={fold+1} fold=======================")
        print("train set size", len(train_idx), len(val_idx))
        split_train_val(train_idx, val_idx)

        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
        #model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)
        print('Model created')

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(params=model.parameters(), 
                                lr=LR_INIT,
                                weight_decay=LR_DECAY)
        print('Optimizer created')

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # 30epochs 마다 1/10 LR 감소
        print('LR Scheduler created')

        print(f" ! Start Training & Validation ! ")
        best_val_acc = 0.0
        for epoch in range(N_EPOCHS):
            train(model, data_loaders['train'], optimizer, criterion, epoch, DEVICE)
            
            val_loss, val_acc = evaluate(model, data_loaders['val'], criterion, DEVICE, BATCH_SIZE, CLASSES, mode='val')
            
            if val_acc > best_val_acc:
                print(f"\nBEST [Epoch {epoch}] Val Loss: {val_loss:.3f}\tTest Accuracy: {val_acc:.3f} % \n")
                save_best_model(model, fold)
                best_val_acc = val_acc
            else: print(f"\n[Epoch {epoch}] Val Loss: {val_loss:.3f}\tVal Accuracy: {val_acc:.3f} % \n")

            lr_scheduler.step()
        print(f"Finished Training & Validation")

        best_model=resnet50().to(DEVICE)
        best_model.load_state_dict(torch.load(f'models/best_{MODEL_NAME}_{fold}Fold.pth'))

        test_loss, test_acc = evaluate(model, data_loaders['test'], criterion, DEVICE, BATCH_SIZE, CLASSES, mode='test')
        print(f"Test Loss : {test_loss}\tTest Accuracy : {test_acc}")
        metrics_df.loc[fold, :] = test_acc

    print(f"10 Fold TEST Acc\n{metrics_df}")
    print(f'Mean test Acc: {metrics_df["acc"].mean():.3f} ({metrics_df["acc"].std():.3f})')