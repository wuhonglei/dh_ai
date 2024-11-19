"""
实现位置编码（Positional Encoding）
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-math.log(10000.0) *
                             torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # 添加 batch_size 维度
        self.register_buffer('pe', pe)  # 不作为模型参数，但会在模型保存和加载时包含

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # 将位置编码添加到输入上
        return self.dropout(x)


if __name__ == '__main__':
    pe = PositionalEncoding(20)
    print(pe.pe.shape)  # torch.Size([5000, 20])
    x = torch.zeros(1, 100, 20)
    y = pe(x)
    print(y.shape)  # torch.Size([1, 100, 20])
