import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from AlexNet import AlexNet
# from tensorboardX import SummaryWriter

# ImageFolder : 각 datasets_type(train, val, test)에 따라 directory에 존재하는 image 메타정보를 tuple로 저장
#                   image 메타 정보 : 이미지 개수, augmentation 기법들(transformation), ...
# dataset_sizes : 각 datasets_type에 따른 data 개수, model accuracy 계산 위해 사용
# image_datasets = {datasets_type: datasets.ImageFolder(root=os.path.join(DATA_DIR, datasets_type), transform=data_transforms[datasets_type]) 
#                   for datasets_type in['train', 'val']}
# dataset_sizes = {datasets_type: len(image_datasets[datasets_type]) 
#                  for datasets_type in ['train', 'val']}
# num_workers : CPU가 GPU에 data 올릴 때 사용할 subprocess 개수(0이면 main process 하나만 사용)
#               batch size가 클 땐 num_workers를 줄여야 함
# inputs : batch단위로 저장된 datasets

def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"train Epoch: {Epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)}({100. * batch_idx / len(train_loader):.0f}%)]\tTrain Loss: {loss.item()}")

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# pytorch device 정의하기
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using PyTorch version: {torch.__version__}, Device: {DEVICE}")

# model parameters 정의하기
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227 # pixels
NUM_CLASSES = 1000
DEVICE_IDS = [0, 1, 2, 3]

DATA_DIR = 'data/'
data_transforms = {
    "train" : transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}

if __name__ == '__main__':
    model = AlexNet(num_classes=NUM_CLASSES).to(DEVICE)
    model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)
    print(model)
    print('Model created')

    image_datasets = {'trainset': datasets.CIFAR10(root=os.path.join(DATA_DIR, 'trainset'), train=True, download=True, transform=data_transforms['train']),
                    'testset': datasets.CIFAR10(root=os.path.join(DATA_DIR, 'testset'), train=False, download=True)}
    print('Dataset created')

    dataloaders = {'trainset': torch.utils.data.DataLoader(image_datasets['trainset'], 
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True, 
                                                           num_workers=8,pin_memory=True,drop_last=True),
               'testset': torch.utils.data.DataLoader(image_datasets['testset'], 
                                                      batch_size=BATCH_SIZE, 
                                                      shuffle=False, 
                                                      num_workers=4)}
    print('Dataloader created')
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY)
    print('Optimizer created')

    # lr_scheduler로 LR 감소시키기 : 30epochs 마다 1/10
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train(model, dataloaders['trainset'], optimizer, log_interval=200)
        test_loss, test_accuracy = evaluate(model, dataloaders['testset'])
        print(f"\n[EPOCH: {epoch+1}]\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy} % \n")



    # # train 시작
    # print('Starting training...')
    # total_steps = 1
    # for epoch in range(NUM_EPOCHS):
    #     lr_scheduler.step()
    #     for imgs, classes in dataloaders['trainset']:
    #         imgs, classes = imgs.to(device), classes.to(device)

    #         # loss 계산
    #         output = model(imgs)
    #         loss = F.cross_entropy(output, classes)

    #         # parameter 갱신
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # log the information and add to tensorboard
    #         # 정보를 기록하고 tensorboard에 추가하기
    #         if total_steps % 10 == 0:
    #             with torch.no_grad():
    #                 _, preds = torch.max(output, 1)
    #                 accuracy = torch.sum(preds == classes)

    #                 print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
    #                     .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
    #                 # tbwriter.add_scalar('loss', loss.item(), total_steps)
    #                 # tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

    #         # gradient values와 parameter average values 추력하기
    #         if total_steps % 100 == 0:
    #             with torch.no_grad():
    #                 # parameters의 grad 출력하고 저장하기
    #                 # parameters values 출력하고 저장하기
    #                 print('*' * 10)
    #                 for name, parameter in model.named_parameters():
    #                     if parameter.grad is not None:
    #                         avg_grad = torch.mean(parameter.grad)
    #                         print('\t{} - grad_avg: {}'.format(name, avg_grad))
    #                         # tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
    #                         # tbwriter.add_histogram('grad/{}'.format(name),
    #                         #         parameter.grad.cpu().numpy(), total_steps)
    #                     if parameter.data is not None:
    #                         avg_weight = torch.mean(parameter.data)
    #                         print('\t{} - param_avg: {}'.format(name, avg_weight))
    #                         # tbwriter.add_histogram('weight/{}'.format(name),
    #                         #         parameter.data.cpu().numpy(), total_steps)
    #                         # tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

    #         total_steps += 1

        # # checkpoints 저장하기
        # checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_states_e{}.pkl'.format(epoch + 1))
        # state = {
        #     'epoch': epoch,
        #     'total_steps': total_steps,
        #     'optimizer': optimizer.state_dict(),
        #     'model': model.state_dict(),
        #     'seed': seed,
        # }
        # torch.save(state, checkpoint_path)