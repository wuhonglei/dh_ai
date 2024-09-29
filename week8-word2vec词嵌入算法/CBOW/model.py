import os
import torch
import torch.nn as nn
import jieba
import nltk
from nltk.corpus import stopwords

from typing import Any


class CBow(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBow, self).__init__()
        # (vocab_size, embedding_dim)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

    def forward(self, x):
        x = x.view(-1, x.size(-1))
        x = self.embeddings(x)
        x = x.mean(dim=1, keepdim=True)
        x = self.linear(x)
        return x.view(x.size(0), -1)
