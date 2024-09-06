import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, captcha_length, class_num):
        super(CNNModel, self).__init__()
        self.captcha_length = captcha_length
        self.class_num = class_num

        """
        Conv2d: 1 * 128 * 128 -> 8 * 128 * 128 -> 8 * 64 * 64
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding='same',
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """
        Conv2d: 8 * 64 * 64 -> 16 * 64 * 64 -> 16 * 32 * 32
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding='same',
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """
        Conv2d: 16 * 32 * 32 -> 16 * 32 * 32 -> 16 * 16 * 16
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding='same',
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """
        Linear: 16 * 16 * 16 -> 128
        """
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU(),
        )

        """
        Linear: 128 -> output_size
        """
        self.fc2 = nn.Sequential(
            nn.Linear(128, self.captcha_length * self.class_num),
        )

    def forward(self, x):
        x = x.view(-1, 1, 128, 128)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits.view(-1, self.captcha_length, self.class_num)


if __name__ == '__main__':
    model = CNNModel(captcha_length=4, class_num=10)

    def print_parameters(model):
        param_count = 0
        for name, param in model.named_parameters():
            param_count += param.numel()
            print(name, param.size())
        print(f'param_count: {param_count}')

    def print_forward(model):
        import torch
        x = torch.randn(1, 1, 128, 128)
        for name, module in model.named_children():
            if name == 'fc1':
                x = x.view(-1, 16 * 16 * 16)
            print(f'name: {name}, input: {x.size()}')
            x = module(x)
            print(f'name: {name}, output: {x.size()}')
            print()

    print(model)
    print_parameters(model)
    # print_forward(model)
