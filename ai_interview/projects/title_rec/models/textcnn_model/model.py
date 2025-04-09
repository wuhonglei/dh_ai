"""
使用 TextCNN 模型进行标题分类
"""

from numpy import dtype
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchinfo import summary


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)
        # x: (batch_size, embedding_dim, sequence_length)
        x = x.permute(0, 2, 1)
        # x_i: (batch_size, num_filters, sequence_length - kernel_size + 1)
        x = [conv(x) for conv in self.convs]
        x = [F.relu(i) for i in x]
        # x_i: (batch_size, num_filters)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x: (batch_size, num_filters * len(filter_sizes))
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 设置参数
    vocab_size = 80000  # 词汇表大小
    embedding_dim = 300  # 词向量维度
    num_filters = 100  # 卷积核数量
    filter_sizes = [3, 4, 5]  # 卷积核大小
    num_classes = 30  # 类别数量

    # 创建模型
    model = TextCNN(vocab_size, embedding_dim,
                    num_filters, filter_sizes, num_classes)

    # 打印模型结构
    # 输入形状为 (batch_size, sequence_length)
    input_data = torch.randint(0, vocab_size, (1, 100), dtype=torch.long)
    summary(model, input_data=input_data)
