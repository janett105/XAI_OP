import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm  # Transformer models library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def main():
    # CIFAR-10 데이터 로드 및 전처리
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT 입력 크기 조정
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 훈련 데이터셋 로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset = torch.utils.data.Subset(trainset, indices=range(0, len(trainset), 100))  # 데이터셋 크기 축소
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    # CIFAR-10 테스트 데이터셋 로드
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testset = torch.utils.data.Subset(testset, indices=range(0, len(testset), 1000))  # 데이터셋 크기 축소
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # ViT 모델 로드 및 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)
    model = model.to(device)

    # 모델 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(1):  # 간단한 데모를 위해 에폭 수는 적게 설정
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')


    print('Finished Training')

    # 결과 이미지 저장 폴더 생성
    output_folder = './vit_grad_cam_results'
    os.makedirs(output_folder, exist_ok=True)

    # 모델 평가 및 Grad-CAM 설정 ##############################################################################
    model.eval()
    grad_cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1])

    # 테스트 데이터 로드 및 
    test_images, test_labels = next(iter(testloader)) #테스트셋에서 배치 로드
    test_images = test_images.to(device)
    
    # 모델 예측 및 가장 확률이 높은 클래스를 대상으로 Grad-CAM 적용
    with torch.no_grad():
        outputs = model(test_images)
        predicted_classes = outputs.argmax(dim=1)  # 가장 높은 확률을 가진 클래스 인덱스
        targets = [ClassifierOutputTarget(predicted_classes[idx]) for idx in range(test_images.size(0))]
        
    cams = grad_cam(test_images, targets=targets)  # Grad-CAM 적용
    ###########################################################################################################
    
    # 이미지 처리 및 저장    
    for idx, cam_image in enumerate(cams):
        img = test_images[idx].cpu().numpy().transpose(1, 2, 0)
        img = (img * np.array((0.5, 0.5, 0.5))) + np.array((0.5, 0.5, 0.5))
        img = np.clip(img, 0, 1)
        cam_image = show_cam_on_image(img, cam_image, use_rgb=True)
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        file_name = f'vit_grad_cam_{idx}.jpg'
        cv2.imwrite(os.path.join(output_folder, file_name), np.uint8(255 * cam_image_bgr))
        print(f'{file_name} has been saved.')

if __name__ == '__main__':
    main()
