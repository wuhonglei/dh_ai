"""
实现解码器（Decoder）
"""

import math
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from DecoderLayer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        output = self.fc_out(tgt)  # [batch_size, tgt_seq_len, tgt_vocab_size]
        return output
