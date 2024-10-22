"""
不使用 RNN 的关键词分类模型
"""

import torch
import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int, dropout: float = 0):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        """
        input: [batch_size, seq_len]
        output: [batch_size, seq_len, embed_dim]
        """
        x = self.embedding(x)
        # [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim]
        x = torch.sum(x, dim=1)
        # [batch_size, embed_dim] -> [batch_size, output_size]
        output = self.fc(x)
        return output
