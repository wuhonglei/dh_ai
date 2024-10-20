import torch
import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.fc(x)
        return x
