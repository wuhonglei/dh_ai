import torch
import torch.nn as nn
from torchinfo import summary


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 第一个卷积层块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二个卷积层块
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(1*576, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # 输入层: 1x32x32
        x = self.conv1(x)  # -> 6x14x14
        x = self.conv2(x)  # -> 16x5x5
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # -> 10
        return x


if __name__ == '__main__':
    # 测试代码
    model = LeNet5(num_classes=10)
    summary(model, input_size=(1, 3, 32, 32))
