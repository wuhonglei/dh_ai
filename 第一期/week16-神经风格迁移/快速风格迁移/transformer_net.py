import torch
import torch.nn as nn

from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        layer1 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),  # 边缘反射填充
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),  # 归一化
            nn.ReLU()
        )

        layer2 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True)
        )

        self.conv_block = nn.Sequential(
            layer1,
            layer2
        )

    def forward(self, x):
        return x + self.conv_block(x)


class TransformerNet(nn.Module):
    def __init__(self, channels=3):
        super(TransformerNet, self).__init__()

        self.conv = nn.Sequential(
            # input: 3 x 256 x 256 output: 32 x 256 x 256
            nn.Sequential(
                nn.ReflectionPad2d(padding=4),
                nn.Conv2d(in_channels=channels, out_channels=32,
                          kernel_size=9, stride=1, bias=False),
                nn.InstanceNorm2d(num_features=32, affine=True),
                nn.ReLU()
            ),
            # input: 32 x 256 x 256 output: 64 x 128 x 128
            nn.Sequential(
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=3, stride=2, bias=False),
                nn.InstanceNorm2d(num_features=64, affine=True),
                nn.ReLU()
            ),
            # input: 64 x 128 x 128 output: 128 x 64 x 64
            nn.Sequential(
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=3, stride=2,  bias=False),
                nn.InstanceNorm2d(num_features=128, affine=True),
                nn.ReLU()
            ),
        )

        # Residual blocks (5 blocks) 128 x 64 x 64
        self.residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.up = nn.Sequential(
            # input: 128 x 64 x 64 output: 64 x 128 x 128
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(num_features=64, affine=True),
                nn.ReLU()
            ),
            # input: 64 x 128 x 128 output: 32 x 256 x 256
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(num_features=32, affine=True),
                nn.ReLU()
            ),
            # input: 32 x 256 x 256 output: 3 x 256 x 256
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=channels,
                          kernel_size=9, stride=1, padding=4, bias=False),
                nn.Tanh()
            )
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        x = self.up(x)
        return x


if __name__ == '__main__':
    channels = 3
    model = TransformerNet(channels)
    summary(model, input_size=(5, channels, 256, 256),
            col_names=("input_size", "output_size", "num_params"))
