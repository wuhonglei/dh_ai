import torch
import torch.nn as nn

from dataset import NamesDataset


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h_reset = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h_update = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h_candidate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        combined = torch.concat((input, hidden), 1)
        reset_gate = torch.sigmoid(self.i2h_reset(combined))
        update_gate = torch.sigmoid(self.i2h_update(combined))

        combined = torch.concat((input, hidden * reset_gate), 1)
        candidate = torch.tanh(self.i2h_candidate(combined))
        hidden = update_gate * hidden + (1 - update_gate) * candidate
        return hidden

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size, device=device)

    def compute_output(self, hidden):
        return self.output(hidden)
