"""
使用 TextCNN 模型进行标题分类
"""

import time
from numpy import dtype
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchinfo import summary


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, padding_idx):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        # x: (batch_size, 1, sequence_length, embedding_dim)
        x = [conv(x) for conv in self.convs]
        # x: (batch_size, num_filters, sequence_length - kernel_size + 1)
        x = [F.relu(i).squeeze(3) for i in x]
        # x_i: (batch_size, num_filters)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x: (batch_size, num_filters * len(filter_sizes))
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 设置参数
    padding_idx = 0  # 填充索引
    vocab_size = 80000  # 词汇表大小
    embedding_dim = 300  # 词向量维度
    num_filters = 100  # 卷积核数量
    filter_sizes = [3, 4, 5]  # 卷积核大小
    num_classes = 30  # 类别数量
    batch_size = 20000  # 批量大小
    sequence_length = 10  # 序列长度
    num_epochs = 1  # 训练轮数

    # 创建模型
    model = TextCNN(vocab_size, embedding_dim,
                    num_filters, filter_sizes, num_classes, padding_idx)

    # 打印模型结构
    # 输入形状为 (batch_size, sequence_length)
    input_data = torch.randint(
        0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
        outputs = model(input_data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Time: {time.time() - start_time}")
    end_time = time.time()
    print(f"Total Time: {end_time - start_time}")
