"""
不使用 RNN 的关键词分类模型
"""

import torch
import torch.nn as nn


class KeywordCategoryModel(nn.Module):
    def __init__(self, input_size: int,  hidden_size: list[int], output_size: int, dropout: float = 0.5):
        super(KeywordCategoryModel, self).__init__()
        self.fc = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(input_size if i ==
                          0 else hidden_size[i-1], hidden_size[i]),
                nn.Tanh(),
                nn.Dropout(dropout)
            ) for i in range(len(hidden_size))
        ])

        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        """
        input: [batch_size, num_features]
        output: [batch_size, output_size]
        """
        x = self.fc(x)
        x = self.output(x)
        return x
