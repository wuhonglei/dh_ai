import torch.nn as nn
from torchinfo import summary


class CRNN(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int):
        """
        Args:
            n_classes: 验证码字符类别数。
            hidden_size: Hidden size of LSTM.
        """
        super(CRNN, self).__init__()
        self.dropout_cnn_rnn = nn.Dropout(0.15)
        self.dropout_fc = nn.Dropout(0.35)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3,
                      stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(
                2, 1), padding=(0, 1)),  # Pool3

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Conv6
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1, 12))  # Adaptive Max Pool
        )
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=hidden_size,
                    num_layers=2, bidirectional=True, batch_first=False, dropout=0.25),
        )
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        b, c, h, w = x.size()  # (batch_size, channels, height, width)
        assert h == 1, "Height must be 1 after CNN layers."
        x = x.squeeze(2)  # (batch_size, channels, width)

        # Transpose for RNN
        x = x.permute(2, 0, 1)  # (width, batch_size, channels)

        x = self.dropout_cnn_rnn(x)

        # RNN
        x, _ = self.rnn(x)

        x = self.dropout_fc(x)

        # Fully connected
        x = self.fc(x)  # (width, batch_size, n_classes)
        return x


# 创建一个字典来存储激活值
activations = {}


def get_activation(name):
    # 定义钩子函数
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def register_hook(model):
    cnn_layers = model.cnn
    cnn_names = []
    for i, layer in enumerate(cnn_layers):
        if isinstance(layer, nn.Conv2d):
            name = f'conv_{i}'
            cnn_names.append(name)
            layer.register_forward_hook(get_activation(name))
    return cnn_names


if __name__ == '__main__':
    img_height = 32
    img_width = 96
    n_channels = 1
    n_classes = 37
    n_hidden = 256
    batch_size = 1

    crnn = CRNN(n_classes, n_hidden)
    print(crnn)
    summary(crnn, input_size=(batch_size, n_channels, img_height, img_width))
