from torch import nn
class Generator(nn.Module):
    def __init__(self, noise_size, n_channels):
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
                        out_channels = n_channels,
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

def print_x_shape(x, model):
    print("print_x_shape:")
    print(f"begin x: {x.shape}")
    x = model.ct1(x)
    print(f"after ct1: {x.shape}")
    x = model.bn1(x)
    print(f"after bn1: {x.shape}")
    x = model.relu1(x)
    print(f"after relu1: {x.shape}")
    x = model.ct2(x)
    print(f"after ct2: {x.shape}")
    x = model.bn2(x)
    print(f"after bn2: {x.shape}")
    x = model.relu2(x)
    print(f"after relu2: {x.shape}")
    x = model.ct3(x)
    print(f"after ct3: {x.shape}")
    x = model.bn3(x)
    print(f"after bn3: {x.shape}")
    x = model.relu3(x)
    print(f"after relu3: {x.shape}")
    x = model.ct4(x)
    print(f"after ct4: {x.shape}")
    x = model.bn4(x)
    print(f"after bn4: {x.shape}")
    x = model.relu4(x)
    print(f"after relu4: {x.shape}")
    x = model.ct5(x)
    print(f"after ct5: {x.shape}")
    x = model.tanh(x)
    print(f"after tanh: {x.shape}")



import torch
from torchvision import utils
if __name__ == '__main__':
    noise_size = 100
    n_channels = 3
    netG = Generator(noise_size, n_channels)
    print(netG)
    fixed_noise = torch.randn(20, noise_size, 1, 1)
    fake = netG(fixed_noise)
    utils.save_image(fake.detach(), 'fake.png', normalize=True)
    print_x_shape(fixed_noise, netG)

