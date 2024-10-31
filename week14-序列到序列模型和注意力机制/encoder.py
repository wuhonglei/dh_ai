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
            embed_size, hidden_size, num_layers, dropout=p, batch_first=True)

    def forward(self, src):
        """
        src: [batch_size, seq_len]
        embedding: [batch_size, seq_len, embed_size]
        """
        embedding = self.dropout(self.embedding(src))

        """
        output: [batch_size, seq_len, hidden_size]
        hidden: [batch_size, num_layers, hidden_size]
        cell: [batch_size, num_layers, hidden_size]
        """
        output, (hidden, cell) = self.rnn(embedding)

        return output, hidden, cell
