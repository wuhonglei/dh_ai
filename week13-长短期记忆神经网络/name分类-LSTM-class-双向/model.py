import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=False, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input: torch.Tensor):
        """
        input shape: (seq_len, 1, input_size)
        lstm output shape: (seq_len, 1, hidden_size * 2)
        """
        _, (hidden, _) = self.lstm(input)
        hidden_forward = hidden[0, :, :]
        hidden_backward = hidden[1, :, :]
        hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)

        # 取最后一个时间步的输出用于分类
        output = self.fc(hidden)
        return output
