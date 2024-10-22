import torch
import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int, dropout: float = 0):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(embed_dim, hidden_size,
                          batch_first=True, bidirectional=False)
        hidden_size *= 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        # hidden: [num_layers * num_directions, batch, hidden_size]
        _, hidden = self.rnn(x)
        x = self.dropout(hidden.squeeze(0))
        output = self.fc(x)
        return output
