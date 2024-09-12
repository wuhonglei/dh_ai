"""
判别器
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    image_size = 784

    netD = Discriminator(image_size, 256, 1)
    image = torch.randn(3, image_size)
    netD.eval()
    with torch.no_grad():
        output = netD(image)
    print(output.shape)
