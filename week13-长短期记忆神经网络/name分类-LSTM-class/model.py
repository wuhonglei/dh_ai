import torch
import torch.nn as nn

from dataset import NamesDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor):
        """
        input shape: (seq_len, 1, input_size)
        """
        _, (hidden, _) = self.lstm(input)
        output = self.fc(hidden.squeeze(0))
        return output

    def init_hidden(self, device):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))
