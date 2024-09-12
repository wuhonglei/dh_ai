import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from utils import print_parameters, print_forward


class Generator(nn.Module):
    def __init__(self, noise_size, output_channels):
        super(Generator, self).__init__()

        """
        生成器的输入是一个噪声向量, 输出是一个图像数据
        noise_size * 1 * 1 -> 1024 * 4 * 4
        """
        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=noise_size,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        """
        1024 * 4 * 4 -> 512 * 8 * 8
        """
        self.ct2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        """
        512 * 8 * 8 -> 256 * 16 * 16
        """
        self.ct3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        """
        256 * 16 * 16 -> 128 * 32 * 32
        """
        self.ct4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        """
        128 * 32 * 32 -> output_channels * 64 * 64
        """
        self.ct5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

        self.ct = nn.Sequential(
            self.ct1,
            self.ct2,
            self.ct3,
            self.ct4,
            self.ct5
        )

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)
        x = self.ct(x)
        return x


if __name__ == '__main__':
    noise_size = 100

    netG = Generator(noise_size, 3)
    noise = torch.randn(20, noise_size)
    output = netG(noise)  # 20 * 3 * 64 * 64
    # print_parameters(netG)
    print_forward(noise, netG)
