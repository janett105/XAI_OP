# 필요한 PyTorch 관련 라이브러리를 불러옵니다.
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def run():
    # 데이터 변환(Transform) 설정: 이미지를 처리하고 모델에 입력하기 전에 적용하는 일련의 작업입니다.
    transform = transforms.Compose([
        transforms.Resize(256),  # 이미지의 크기를 256x256으로 조정합니다.
        transforms.CenterCrop(224),  # 이미지 중앙에서 224x224 크기로 자릅니다.
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환합니다.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화: 각 채널의 색상 값을 조정합니다.
    ])

    # CIFAR10 훈련 데이터셋을 다운로드하고, 변환(transform)을 적용합니다.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 데이터로더를 생성: 훈련 데이터셋을 배치 단위로 나누고, 셔플하여 모델 학습 시 제공합니다.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # CIFAR10 테스트 데이터셋을 다운로드하고, 변환(transform)을 적용합니다.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # 데이터로더를 생성: 테스트 데이터셋을 배치 단위로 나누어 모델 평가 시 제공합니다.
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # CIFAR10 데이터셋의 클래스(레이블)을 정의합니다.
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 훈련 데이터셋과 테스트 데이터셋을 로드합니다.(이미지넷 활용시)
    #train_dataset = datasets.ImageFolder(root='path/to/train', transform=transform)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    #test_dataset = datasets.ImageFolder(root='path/to/val', transform=transform)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 모델 정의: ResNet18을 사용하고, 마지막 레이어를 CIFAR10의 클래스 수(10)에 맞게 조정합니다.
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(classes))  # 모델의 출력 레이어를 10개 클래스에 맞게 설정합니다.

    # 손실 함수와 최적화 알고리즘을 정의합니다.
    criterion = nn.CrossEntropyLoss()  # 분류 문제에 사용되는 교차 엔트로피 손실 함수입니다.
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD 최적화 알고리즘을 사용하며, 학습률과 모멘텀을 설정합니다.

    # 학습 과정: 설정한 에폭만큼 반복하여 모델을 학습시킵니다.
    for epoch in range(2):  # 전체 데이터셋을 두 번 반복합니다.
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # 데이터로부터 입력 이미지와 레이블을 가져옵니다.

            optimizer.zero_grad()  # 이전 반복에서의 그라디언트를 초기화합니다.

            # 순전파: 모델에 입력을 넣고 출력을 계산합니다.
            outputs = net(inputs)
            # 손실 계산: 모델 출력과 실제 레이블 간의 손실을 계산합니다.
            loss = criterion(outputs, labels)
            # 역전파: 손실에 따라 그라디언트를 계산합니다.
            loss.backward()
            # 최적화: 계산된 그라디언트를 사용하여 모델의 파라미터를 조정합니다.
            optimizer.step()

            # 출력 로그: 현재 손실을 누적하고, 일정 간격으로 평균 손실을 출력합니다.
            running_loss += loss.item()
            if i % 2000 == 1999:    # 매 2000 미니배치마다 평균 손실을 출력합니다.
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('학습 완료')  # 모든 에폭의 학습이 끝났음을 알립니다.

    # 학습이 완료된 모델 저장
    torch.save(net.state_dict(), 'model_trained.pth')
    print('모델 저장 완료')
    
if __name__ == '__main__':
    run()  # 위에서 정의한 run 함수를 실행하여 학습을 시작합니다.
