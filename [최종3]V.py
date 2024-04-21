import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import random
import os

# 데이터셋 로드 및 서브셋 생성
def load_subset_of_cifar10(subset_fraction=1/1000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    indices_train = random.sample(range(len(train_dataset)), int(len(train_dataset) * subset_fraction))
    indices_test = random.sample(range(len(test_dataset)), int(len(test_dataset) * subset_fraction))

    subset_train = Subset(train_dataset, indices_train)
    subset_test = Subset(test_dataset, indices_test)

    return subset_train, subset_test

# 모델 로드 및 GradCAM 설정
def configure_model_and_cam():
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model.eval()
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    return model, cam

# reshape_transform 함수
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 폴더 생성
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# CAM 이미지 생성 및 저장
def generate_and_save_cam_images(data_loader, model, cam, output_folder):
    create_directory(output_folder)  # 폴더 생성
    for i, (images, _) in enumerate(data_loader):
        rgb_img = np.float32(images.squeeze().numpy().transpose((1, 2, 0)))
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cv2.imwrite(f'{output_folder}/cam_image_{i}.jpg', cam_image * 255)

if __name__ == '__main__':
    train_data, test_data = load_subset_of_cifar10()
    model, cam = configure_model_and_cam()
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    output_folder = './cam_images'  # 이미지 저장 폴더

    generate_and_save_cam_images(train_loader, model, cam, output_folder)
    generate_and_save_cam_images(test_loader, model, cam, output_folder)
