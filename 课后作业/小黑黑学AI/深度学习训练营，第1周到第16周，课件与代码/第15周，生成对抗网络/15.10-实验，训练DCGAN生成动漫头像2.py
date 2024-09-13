from torch import nn
import torch
from torchvision import utils

class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.ct1 = nn.ConvTranspose2d(
                        in_channels = noise_size,
                        out_channels = 1024,
                        kernel_size = 4,
                        stride = 1,
                        padding = 0,
                        bias = False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.ct2 = nn.ConvTranspose2d(
                        in_channels = 1024,
                        out_channels = 512,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)

        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()

        self.ct3 = nn.ConvTranspose2d(
                        in_channels = 512,
                        out_channels = 256,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.ct4 = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels = 128,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.ct5 = nn.ConvTranspose2d(
                        in_channels = 128,
                        out_channels = 3,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.ct1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.ct2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.ct3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.ct4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.ct5(x)
        x = self.tanh(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    netG = Generator(100)
    netG.load_state_dict(torch.load('anime-face.pth'))
    netG.eval()

    fixed_noise = torch.randn(64, 100, 1, 1)

    fake = netG(fixed_noise)
    fake = utils.make_grid(fake, padding=2, normalize=True)
    plt.imshow(fake.permute(1, 2, 0).numpy())
    plt.show()
