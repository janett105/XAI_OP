import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def main():
    # 데이터 로드 및 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델의 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 데이터셋의 크기를 축소하여 사용
    trainset = torch.utils.data.Subset(trainset, indices=range(0, len(trainset), 20)) #1/20
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)  # num_workers 조정

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)

    # ViT 모델 정의
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=10)
    model.to(device)

    # 모델 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # 에폭 수 감소
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 출력 빈도 조정
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 테스트셋에서 이미지 하나를 불러오기
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    img = images[0].unsqueeze(0).to(device)

    # Grad-CAM 적용
    target_layer = model.blocks[-1].norm1  # 모델의 마지막 self-attention layer를 대상으로 설정
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img, targets=None)  # 타겟 클래스 지정 없이 CAM 계산

    # 시각화
    grayscale_cam = grayscale_cam[0, :]
    img = img.cpu().data.numpy()
    img = img.squeeze().transpose(1, 2, 0)
    img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    img = np.clip(img, 0, 1)

    # CAM 결과와 원본 이미지 함께 표시
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(cam_image)
    axs[1].set_title('Grad-CAM')
    axs[1].axis('off')
    plt.show()

if __name__ == '__main__':
    main()
