import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor):
        """
        input shape: (seq_len, 1, input_size)
        """
        _, hidden = self.gru(input)
        output = self.fc(hidden.squeeze(0))
        return output
