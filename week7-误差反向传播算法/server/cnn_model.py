import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input channels, 6 output channels, 5x5 kernel
        self.pool1 = nn.MaxPool2d(2, 2)

        # 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 input channels, 16 output channels, 5x5 kernel
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)  # input shape: (batch_size, 1, 28, 28)
        x = torch.relu(x)  # output shape: (batch_size, 6, 24, 24)
        x = self.pool1(x)  # output shape: (batch_size, 6, 12, 12)

        x = self.conv2(x)  # input shape: (batch_size, 6, 12, 12)
        x = torch.relu(x)  # output shape: (batch_size, 16, 8, 8)
        x = self.pool2(x)  # output shape: (batch_size, 16, 4, 4)

        x = x.view(x.shape[0], -1)  # output shape: (batch_size, 16*4*4)
        x = self.fc1(x)  # output shape: (batch_size, 120)
        x = torch.relu(x)
        x = self.fc2(x)  # output shape: (batch_size, 84)
        x = torch.relu(x)
        x = self.fc3(x)  # output shape: (batch_size, 10)

        return x

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            prob = torch.softmax(output, dim=1)
            prob_list = prob.squeeze()
            prob, index = torch.max(prob, 1)

        return index, prob, prob_list
