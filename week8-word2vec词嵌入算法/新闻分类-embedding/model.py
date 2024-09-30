import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """
    新闻文本分类器
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, padding_idx):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim], seq_len 表示句子长度
        x = self.embedding(x)
        x = torch.sum(x, dim=1)  # x: [batch_size, embedding_dim]
        x = self.classifier(x)
        return x
