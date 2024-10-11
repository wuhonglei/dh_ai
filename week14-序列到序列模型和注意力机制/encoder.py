import torch
import torch.nn as nn

from torchinfo import summary


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(
            embed_size, hidden_size, num_layers, dropout=p)

    def forward(self, src):
        """
        src: [seq_len, batch_size]
        embedding: [seq_len, batch_size, embed_size]
        """
        embedding = self.dropout(self.embedding(src))

        """
        hidden: [num_layers, batch_size, hidden_size]
        cell: [num_layers, batch_size, hidden_size]
        """
        _, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


if __name__ == '__main__':
    input_size = 100  # 词典大小
    embed_size = 50  # 词向量维度
    hidden_size = 1024  # 隐藏层维度
    num_layers = 2  # LSTM层数
    p = 0.5  # dropout概率

    encoder = Encoder(input_size, embed_size, hidden_size, num_layers, p)
    input = torch.randint(0, input_size, (10, 1))
    hidden, cell = encoder(input)
    summary(encoder, input_size=(10, 1), dtypes=[torch.long], device='cpu', col_names=[
            "input_size", "output_size", "num_params"])
    print(hidden.shape, cell.shape)
