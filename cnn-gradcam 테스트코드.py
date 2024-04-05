import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# CIFAR-10 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 모델 학습
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # 데이터셋을 여러 번 반복
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # 매 200 미니배치마다 출력
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# 테스트셋에서 이미지 하나를 불러오기
dataiter = iter(testloader)
images, labels = next(dataiter)

# 이미지 하나 선택
img = images[0].unsqueeze(0).to(device)  # 모델 입력에 맞게 차원 추가 및 장치에 올리기

# Grad-CAM 적용
target_layer = model.conv2
cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=img, targets=None)

# 시각화 준비
grayscale_cam = grayscale_cam[0, :]
img = img.cpu().data.numpy()
img = img.squeeze().transpose(1, 2, 0)
img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
img = np.clip(img, 0, 1)

# CAM 결과 시각화
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# 원본 이미지 시각화 준비
# 이미지를 [0, 1] 범위로 정규화합니다.
original_img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
original_img = np.clip(original_img, 0, 1)

# 원본 이미지와 CAM 이미지를 함께 표시
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1행 2열의 서브플롯 생성
axs[0].imshow(original_img)  # 원본 이미지 표시
axs[0].title.set_text('Original Image')
axs[0].axis('off')  # 축 정보 제거

axs[1].imshow(cam_image)  # CAM 이미지 표시
axs[1].title.set_text('Grad-CAM')
axs[1].axis('off')  # 축 정보 제거

plt.show()  # 플롯 표시
plt.imshow(cam_image)
plt.show()
