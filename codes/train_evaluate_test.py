import torch
import torchvision
from codes.Vizualization import imageshow

def train(model, train_loader, optimizer, criterion, epoch, DEVICE):
    model.train()
    train_loss=0.0

    # 한 epoch당 batch 단위로 전체 train dataset training
    # 1000 batch씩 묶어서 train_loss 평균 print
    for batch_idx, (data_batch, labels) in enumerate(train_loader):
        data_batch = data_batch.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad() # parameter gradients 0으로 초기화

        outputs = model(data_batch)
        loss = criterion(outputs, labels)
        
        train_loss = train_loss + loss.item()
        if (batch_idx+1) % 100 == 0:
            print(f"\nEpoch {epoch} [{(batch_idx+1) * len(data_batch)}/{len(train_loader.dataset)}] Train Loss : {train_loss/1000:.3f}")
            train_loss = 0.0

        loss.backward()
        optimizer.step()
    
def evaluate(model, valtest_loader, criterion, DEVICE, BATCH_SIZE, CLASSES, mode):
    model.eval()
    valtest_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data_batch, labels) in enumerate(valtest_loader):
            data_batch = data_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # test인 경우 한 개의 batch는 gt, prediction 비교 시각화
            if mode=='test' and batch_idx==0:
                imageshow(torchvision.utils.make_grid(data_batch))
                print('Real labels :', ' '.join(f'{CLASSES[labels[j]]}' for j in range(BATCH_SIZE)))

                outputs = model(data_batch)
                prediction = outputs.max(1, keepdim=True)[1]
                print('Predicted labels :', ' '.join(f'{CLASSES[prediction[j]]}' for j in range(BATCH_SIZE)))
            
            outputs = model(data_batch)
            prediction = outputs.max(1, keepdim=True)[1]

            valtest_loss += criterion(outputs, labels).item()
            correct += (prediction == labels).sum().item()

    valtest_loss /= len(valtest_loader.data_batchset)
    valtest_acc = 100. * correct / len(valtest_loader.dataset)
    return valtest_loss, valtest_acc