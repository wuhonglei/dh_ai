import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        """
        input shape: (batch_size, 1, 28, 28)
        conv1: (1, 28, 28) -> (6, 24, 24) -> (6, 12, 12)
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        input shape: (batch_size, 6, 12, 12)
        conv2: (6, 12, 12) -> (16, 8, 8) -> (16, 4, 4)
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        input shape: (batch_size, 16, 4, 4)
        fc1: (16, 4, 4) -> (120,)
        """
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
        )

        """
        input shape: (batch_size, 120)
        fc2: (120,) -> (84,)
        """
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        """
        input shape: (batch_size, 84)
        fc3: (84,) -> (10,)
        """
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        input shape: (batch_size, 1, 28, 28)
        """
        x = x.view(-1, 1, 28, 28)

        """
        input shape: (batch_size, 1, 28, 28)
        conv1: (1, 28, 28) -> (6, 24, 24) -> (6, 12, 12)
        """
        x = self.conv1(x)

        """
        input shape: (batch_size, 6, 12, 12)
        conv2: (6, 12, 12) -> (16, 8, 8) -> (16, 4, 4)
        """
        x = self.conv2(x)

        """
        flatten
        input shape: (batch_size, 16, 4, 4)
        output shape: (batch_size, 16*4*4)
        """
        x = x.view(x.size(0), 16 * 4 * 4)

        """
        input shape: (batch_size, 16*4*4)
        fc1: (16*4*4) -> (120,)
        """
        x = self.fc1(x)

        """
        input shape: (batch_size, 120)
        fc2: (120,) -> (84,)
        """
        x = self.fc2(x)

        """
        input shape: (batch_size, 84)
        fc3: (84,) -> (10,)
        """
        output = self.fc3(x)

        return output

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            prob = torch.softmax(output, dim=1)
            prob_list = prob.squeeze().tolist()
            prob, index = torch.max(prob, 1)
        return index, prob, prob_list


def print_params(model):
    count = 0
    for name, param in model.named_parameters():
        print(f'name: {name}, param: {param.numel()}')
        count += param.numel()
        print(name, param.shape)
    print('total parameters:', count)


def print_forward(model, x):
    print('input x to model:', x.shape)
    for name, layer in model.named_children():
        if name == 'fc1':
            x = x.view(x.shape[0], -1)
        x = layer(x)
        print(f'{name} after size: {x.shape}')


if __name__ == '__main__':
    test_model = LeNet()
    print(test_model)
    print('---')
    print_params(test_model)

    x = torch.randn(64, 1, 28, 28)
    print_forward(test_model, x)
