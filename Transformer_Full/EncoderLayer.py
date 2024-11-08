import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask=None):
        # 自注意力子层
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(src2))

        # 前馈网络子层
        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout(src2))

        return src
