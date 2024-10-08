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
        """
        _, (hidden, _) = self.lstm(input)
        # 合并前向和后向的 hidden state
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=-1)
        output = self.fc(hidden_cat)
        return output
