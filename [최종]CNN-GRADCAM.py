import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import os

def main():
    # CIFAR-10 데이터 로드 및 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 훈련 및 테스트 데이터셋 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, indices=range(0, len(trainset), 50))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset = torch.utils.data.Subset(testset, indices=range(0, len(testset), 1000))
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # 모델 정의
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
    print(f"Using device: {device}")

    # 모델 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.3f}')

    print('Finished Training')

    # 결과 이미지 저장 폴더 생성
    output_folder = './grad_cam_results'
    os.makedirs(output_folder, exist_ok=True)

    # Grad-CAM 적용 및 결과 저장
    target_layer = model.conv2
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 모든 테스트 이미지에 대해 반복
    file_count = 0
    for idx, (images, labels) in enumerate(testloader):
        images = images.to(device)
        grayscale_cams = cam(input_tensor=images, targets=None)

        for img_idx, cam_img in enumerate(grayscale_cams):
            img = images[img_idx].unsqueeze(0).cpu().data.numpy().squeeze().transpose(1, 2, 0)
            img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
            img = np.clip(img, 0, 1)
            cam_image = show_cam_on_image(img, cam_img, use_rgb=True)
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            file_name = f'grad_cam_result_{idx * testloader.batch_size + img_idx}.jpg'
            cv2.imwrite(os.path.join(output_folder, file_name), np.uint8(255 * cam_image_bgr))
            print(f'{file_name} has been saved.')
            file_count += 1

    print(f'All {file_count} Grad-CAM images have been saved.')

if __name__ == '__main__':
    main()
