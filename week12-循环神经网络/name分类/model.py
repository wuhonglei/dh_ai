import torch
import torch.nn as nn

from dataset import NamesDataset


class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        """
        input to hidden
        Linear 层，将输入和隐藏状态连接在一起
        """
        self.i2h = nn.Linear(input_size + hidden_size,
                             hidden_size)
        self.activation = nn.ReLU()
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        """
        input: [1, input_size]
        hidden: [1, hidden_size]
        将输入和隐藏状态连接在一起，传入一个线性层
        """
        combined = torch.concat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.activation(hidden)
        return hidden

    def compute_output(self, hidden: torch.Tensor):
        return self.h2o(hidden)

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size, dtype=torch.float32, device=device)


if __name__ == '__main__':
    word = 'Albert'
    dataset = NamesDataset('data/names')
    word_tensor = dataset.name_to_tensor(word)
    input_size = len(dataset.all_letters)
    hidden_size = 128
    output_size = dataset.get_labels_num()
    model = RNNModel(input_size, hidden_size, output_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_hidden = model.init_hidden(device)
    sequence_len = word_tensor.size(0)
    print('sequence shape:', word_tensor.shape)
    for i in range(sequence_len):
        output, last_hidden = model(word_tensor[i], last_hidden)
        print(f'i: {i} current char: {word[i]}')
        print(f'input shape: {word_tensor[i].shape}')
        print(f'output shape: {output.shape}')
        print(f'hidden shape: {last_hidden.shape}')
        print('-------------------')
