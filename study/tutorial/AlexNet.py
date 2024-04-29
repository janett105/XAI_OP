import torch.nn as nn

'''
input layer(BATCH_SIZE(BS), N_IN_CHANNEL, IMAGE_DIM, IMAGE_DIM)
BS = 3
IMAGE_DIM = 224
transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

conv1 : (BS, 3, 224, 224) -> (BS, 96, 55, 55)
ReLU
Maxpool1 : (BS, 96, 55, 55) -> (BS, 96, 27, 27)
Norm1

conv2 : (BS, 96, 27, 27) -> (BS, 256, 27, 27)
ReLU
Maxpool2 : (BS, 256, 27, 27) -> (BS, 256, 13, 13)
Norm2

conv3 : (BS, 256, 13, 13) -> (BS, 384, 13, 13)
ReLU
conv4 : (BS, 384, 13, 13) -> (BS, 384, 13, 13)
ReLU
conv5 : (BS, 384, 13, 13) -> (BS, 256, 13, 13)
ReLU
Maxpool3 : (BS, 256, 13, 13) -> (BS, 256, 6, 6)

FC1: (BS, 256, 6, 6) -> 4096
FC2 : 4096 -> 4096

output layer : 4096->num_classes
'''
class AlexNet(nn.Module):
    def __init__(self, N_IN_CHANNELS, N_CLASSES):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=N_IN_CHANNELS, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), # inplace=True를 사용하면 gradient계산 전에(loss.backward) loss값이 inplace돼서 오류
            nn.MaxPool2d(kernel_size=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, N_CLASSES),
        )

    #     self.init_bias()

    # def init_bias(self):
    #     for layer in self.net:
    #         if isinstance(layer, nn.Conv2d):
    #             # weight와 bias 초기화
    #             nn.init.normal_(layer.weight, mean=0, std=0.01)
    #             nn.init.constant_(layer.bias, 0)
    #     # 논문에 2,4,5 conv2d layer의 bias는 1로 초기화한다고 나와있습니다.  
    #     nn.init.constant_(self.net[4].bias, 1)
    #     nn.init.constant_(self.net[10].bias, 1)
    #     nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), 256 * 2 * 2) 
        return self.classifier(x)