import torch
import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(0.25)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            batch_first=True, bidirectional=True)
        hidden_size *= 2 if self.lstm.bidirectional else 1
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        # hidden: [num_layers * num_directions, batch, hidden_size]
        _, (hidden, _) = self.lstm(x)
        concat_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        output = self.fc(concat_hidden)
        return output
