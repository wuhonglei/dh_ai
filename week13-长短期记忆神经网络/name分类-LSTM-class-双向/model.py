import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_forward = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.lstm_backward = nn.LSTM(
            input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input: torch.Tensor):
        """
        input shape: (seq_len, 1, input_size)
        """
        _, (hidden_forward, _) = self.lstm_forward(input)
        reversed_input = torch.flip(input, [0])
        _, (hidden_backward, _) = self.lstm_backward(reversed_input)

        hidden = torch.cat(
            (hidden_forward.squeeze(0), hidden_backward.squeeze(0)), dim=-1)
        output = self.fc(hidden)
        return output
