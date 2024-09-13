from torch import nn

class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
                        in_channels = n_channels,
                        out_channels = 64,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
                        in_channels = 64,
                        out_channels = 128,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(
                        in_channels = 128,
                        out_channels = 256,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 512,
                        kernel_size = 4,
                        stride = 2,
                        padding= 1,
                        bias = False)
        self.bn4 = nn.BatchNorm2d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(
                        in_channels = 512,
                        out_channels = 1,
                        kernel_size = 4,
                        stride = 1,
                        padding = 0,
                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)

        x = self.conv5(x)
        x = self.sigmoid(x)
        x = x.view(-1, 1).squeeze(1)
        return x


def print_x_shape(x, model):
    print("print_x_shape:")
    print(f"begin x: {x.shape}")
    x = model.conv1(x)
    print(f"after conv1: {x.shape}")
    x = model.bn1(x)
    print(f"after bn1: {x.shape}")
    x = model.leaky_relu1(x)
    print(f"after leaky_relu1: {x.shape}")
    x = model.conv2(x)
    print(f"after conv2: {x.shape}")
    x = model.bn2(x)
    print(f"after bn2: {x.shape}")
    x = model.leaky_relu2(x)
    print(f"after leaky_relu2: {x.shape}")
    x = model.conv3(x)
    print(f"after conv3: {x.shape}")
    x = model.bn3(x)
    print(f"after bn3: {x.shape}")
    x = model.leaky_relu3(x)
    print(f"after leaky_relu3: {x.shape}")
    x = model.conv4(x)
    print(f"after conv4: {x.shape}")
    x = model.bn4(x)
    print(f"after bn4: {x.shape}")
    x = model.leaky_relu4(x)
    print(f"after leaky_relu4: {x.shape}")
    x = model.conv5(x)
    print(f"after conv5: {x.shape}")
    x = model.sigmoid(x)
    print(f"after sigmoid: {x.shape}")
    x = x.view(-1, 1).squeeze(1)
    print(f"after view: {x.shape}")

import torch

if __name__ == '__main__':
    netD = Discriminator(3)
    print(netD)
    real = torch.randn(5, 3, 64, 64)
    output = netD(real)
    print(f"output shape = {output.shape}")
    print(output)
    print_x_shape(real, netD)

