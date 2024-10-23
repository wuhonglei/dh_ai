"""
不使用 RNN 的关键词分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int, hidden_size: int, sub_category_size: int, num_classes: int, dropout: float = 0.0):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim + sub_category_size, hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(embed_dim + sub_category_size, num_classes)

    def forward(self, word_input: torch.Tensor, sub_category_input: torch.Tensor):
        """
        word_input: [batch_size, seq_len]
        word_output: [batch_size, seq_len, embed_dim]
        sub_category_input: [batch_size, sub_category_size]
        """
        word_embed = self.embedding(word_input)
        word_embed = word_embed.mean(dim=1)
        # # 按行进行 Min-Max 归一化
        # min_vals, _ = torch.min(
        #     word_embed, dim=1, keepdim=True)  # [batch_size, 1]
        # max_vals, _ = torch.max(
        #     word_embed, dim=1, keepdim=True)  # [batch_size, 1]
        # ranges = max_vals - min_vals
        # ranges = torch.where(ranges > 0, ranges,
        #                      torch.ones_like(ranges))  # 避免除以零
        # normalized_tensor = (word_embed - min_vals) / \
        #     ranges  # [batch_size, hidden_size]
        # # [batch_size, embed_dim + sub_category_size]
        x = torch.cat([word_embed, sub_category_input], dim=1)
        x = self.dropout(x)
        # x = self.fc1(x)
        # x = self.activation(x)
        x = self.classifier(x)
        return x
