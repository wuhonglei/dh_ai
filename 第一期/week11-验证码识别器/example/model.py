import torch
import torch.nn as nn
from torchinfo import summary


class CNN_CTC_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_CTC_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入为灰度图像，通道数为 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 高度和宽度均减半

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 只在高度方向池化

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 只在高度方向池化

            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # 自适应平均池化
        )
        self.fc = nn.Linear(512, num_classes)  # 将特征映射到类别数

    def forward(self, x):
        x = self.cnn(x)  # x 的形状为 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        # 调整形状以匹配 CTC 的输入要求
        x = x.permute(3, 0, 2, 1)  # 形状变为 (width, batch_size, height, channels)
        # 展平成 (width, batch_size, feature_size)
        x = x.reshape(width, batch_size, -1)
        x = self.fc(x)  # 计算每个时间步的输出，形状为 (width, batch_size, num_classes)
        x = nn.functional.log_softmax(x, dim=2)  # 计算 log_softmax，用于 CTC 损失
        return x


if __name__ == '__main__':
    pass
    model = CNN_CTC_Model(num_classes=10)
    print(model)
    summary(model, input_size=(1, 1, 64, 128),
            device='cpu', dtypes=[torch.float32])
