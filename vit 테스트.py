import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import timm  # timm 라이브러리를 사용해 다양한 이미지 모델을 활용 가능

def run():
    # 데이터 변환(Transform) 설정
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CIFAR10 데이터셋 로딩
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ViT 모델 정의 및 분류기 조정
    net = timm.create_model('vit_base_patch16_224', pretrained=True)
    net.head = nn.Linear(net.head.in_features, len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 학습 과정
    for epoch in range(2):  # 예시로 2 에폭만 진행
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('학습 완료')

    # 모델 저장
    torch.save(net.state_dict(), 'model_trained.pth')
    print('모델 저장 완료')

    # 테스트 데이터셋에 대한 모델 성능 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'테스트 데이터셋에 대한 모델의 정확도: {100 * correct / total}%')

if __name__ == '__main__':
    run()
