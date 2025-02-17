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

        # 计算重置门和更新门
        r_t = torch.sigmoid(self.i2h_reset(combined))
        z_t = torch.sigmoid(self.i2h_update(combined))

        # 计算候选隐藏状态
        combined_reset = torch.concat((input, hidden * r_t), 1)
        n_t = torch.tanh(self.i2h_candidate(combined_reset))
        hidden = (1 - z_t) * hidden + z_t * n_t
        return hidden

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size, device=device)

    def compute_output(self, hidden):
        return self.output(hidden)
