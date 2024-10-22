"""
不使用 RNN 的关键词分类模型
"""

import torch
import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, input_size: int,  hidden_size1: int, hidden_size2: int, output_size: int):
        super(KeywordCategoryModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.output = nn.Linear(hidden_size1, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        input: [batch_size, num_features]
        output: [batch_size, output_size]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.output(x)
        return x
