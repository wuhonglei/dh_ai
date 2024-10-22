"""
不使用 RNN 的关键词分类模型
"""

import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        output = self.fc(x)
        return output
