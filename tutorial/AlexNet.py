import torch.nn as nn

'''
input layer(batch_size(BS), 3 kernels, 227, 227)

conv1 : (BS, 3, 227, 227) -> (BS, 96, 55, 55)
Maxpool1 : (BS, 96, 55, 55) -> (BS, 96, 27, 27)
Norm1 : (BS, 96, 27, 27) -> (BS, 96, 27, 27)

conv2 : (BS, 96, 27, 27) -> (BS, 256, 27, 27)
Maxpool2 : (BS, 256, 27, 27) -> (BS, 256, 13, 13)
Norm2 : (BS, 256, 13, 13) -> (BS, 256, 13, 13)

conv3 : (BS, 256, 13, 13) -> (BS, 384, 13, 13)
conv4 : (BS, 384, 13, 13) -> (BS, 384, 13, 13)
conv5 : (BS, 384, 13, 13) -> (BS, 256, 13, 13)
Maxpool3 : (BS, 256, 13, 13) -> (BS, 256, 6, 6)

FC1: (BS, 256, 6, 6) -> 4096
FC2 : 4096 -> 4096

output layer : 4096->num_classes
'''
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )
        self.init_bias()

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                # weight와 bias 초기화
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # 논문에 2,4,5 conv2d layer의 bias는 1로 초기화한다고 나와있습니다.  
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        print('전전 : ', x.size())
        x = self.net(x)
        print('전 : ',x.size())
        x = x.view(x.size(0), 256 * 6 * 6) #tensor 형 변환 
        x = self.classifier(x)
        print('후 : ',x.size())
        return x
