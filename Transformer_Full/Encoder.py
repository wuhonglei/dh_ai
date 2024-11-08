"""
实现编码器（Encoder）
"""


import math
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)

        for layer in self.layers:
            src = layer(src, src_mask)
        return src  # [batch_size, seq_len, d_model]
