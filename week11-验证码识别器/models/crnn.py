import torch.nn as nn
from torchinfo import summary


class CRNN(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, n_classes: int):
        """
        Args:
            n_classes: 验证码字符类别数。
            hidden_size: Hidden size of LSTM.
        """
        super(CRNN, self).__init__()
        self.dropout_cnn_rnn = nn.Dropout(0.15)
        self.dropout_fc = nn.Dropout(0.35)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
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
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool3

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
        x = x.permute(3, 0, 1, 2)  # (batch_size, width, channels, height)
        x = x.view(w, b, c * h)  # (batch_size, width, channels * height)

        x = self.dropout_cnn_rnn(x)

        # RNN
        x, _ = self.rnn(x)

        x = self.dropout_fc(x)

        # Fully connected
        x = self.fc(x)  # (width, batch_size, n_classes)
        return x


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from utils import load_config
    config = load_config('../config.yaml')
    model_config = config['model']
    dataset_config = config['dataset']

    img_height = model_config['height']
    img_width = model_config['width']
    n_classes = len(dataset_config['characters']) + 1
    n_hidden = model_config['hidden_size']
    in_channels = model_config['in_channels']
    batch_size = 1

    crnn = CRNN(in_channels, n_hidden, n_classes)
    summary(crnn, input_size=(batch_size, in_channels, img_height, img_width), col_names=(
        "input_size", "output_size", "num_params"))
