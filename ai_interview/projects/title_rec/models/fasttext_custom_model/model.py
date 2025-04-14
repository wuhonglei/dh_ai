"""
使用 FastText 模型进行标题分类
"""

import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FastText(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int, num_classes: int, dropout: float, use_relu: bool):
        """
        FastText 模型实现

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度
            padding_idx: 填充索引
            num_classes: 分类数量
            dropout: dropout 概率，默认为 0.5
        """
        super(FastText, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if use_relu else nn.Identity()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, sequence_length)

        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        # 计算非填充位置的数量
        non_pad_mask = x != self.padding_idx
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        # 使用非填充位置的数量作为分母进行平均池化
        x = x.sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 设置参数
    padding_idx = 0  # 填充索引
    vocab_size = 80000  # 词汇表大小
    embedding_dim = 300  # 词向量维度
    num_classes = 30  # 类别数量
    batch_size = 20000  # 批量大小
    sequence_length = 10  # 序列长度
    num_epochs = 1  # 训练轮数
    dropout = 0.5
    use_relu = True

    # 创建模型
    model = FastText(vocab_size, embedding_dim, padding_idx,
                     num_classes, dropout, use_relu)

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
