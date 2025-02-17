"""
判别器
"""

import torch
import torch.nn as nn
from utils import print_parameters, print_forward


class Discriminator(nn.Module):
    """
    判别器的输入是一个图像数据, 输出是一个概率值
    """

    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        # 3 * 64 * 64 -> 64 * 32 * 32
        self.cv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        # 64 * 32 * 32 -> 128 * 16 * 16
        self.cv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # 128 * 16 * 16 -> 256 * 8 * 8
        self.cv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # 256 * 8 * 8 -> 512 * 4 * 4
        self.cv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # 512 * 4 * 4 -> 1 * 1 * 1
        self.cv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, x.size(1), 64, 64)
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = self.cv5(x)
        return x.view(-1, 1).squeeze(1)


if __name__ == '__main__':
    input_channels = 3
    netD = Discriminator(input_channels)
    image = torch.randn(10, input_channels, 64, 64)
    netD.eval()

    with torch.no_grad():
        output = netD(image)

    # print_parameters(netD)
    print_forward(image, netD)
