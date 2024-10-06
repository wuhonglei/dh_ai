import torch
import torch.nn as nn

from dataset import NamesDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h_forget = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h_input = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h_output = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h_cell = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        combined = torch.concat((input, hidden), 1)
        forget_gate = torch.sigmoid(self.i2h_forget(combined))
        input_gate = torch.sigmoid(self.i2h_input(combined))
        output_gate = torch.sigmoid(self.i2h_output(combined))

        new_cell: torch.Tensor = forget_gate * cell + input_gate * \
            torch.tanh(self.i2h_cell(combined))
        hidden = output_gate * torch.tanh(new_cell)
        return hidden, new_cell

    def init_hidden(self, device):
        return (torch.zeros(1, self.hidden_size, device=device), torch.zeros(1, self.hidden_size, device=device))

    def compute_output(self, hidden):
        return self.output(hidden)
