"""
实现缩放点积注意力（Scaled Dot-Product Attention）
"""

import torch
import math
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    '''
    计算缩放点积注意力
    Q, K, V: [batch_size, num_heads, seq_len, d_k]
    mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
    '''
    d_k = Q.size(-1)
    # [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
    return output, attn


def create_pad_mask(seq, pad_idx):
    """
    序列掩码（Padding Mask）
    用于在注意力计算中屏蔽填充（padding）位置
    """
    # [batch_size, 1, 1, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_subsequent_mask(size):
    """
    未来位置掩码（Subsequent Mask）
    用于在解码器中屏蔽未来的位置，防止信息泄露。
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()  # 上三角矩阵
    return mask  # [seq_len, seq_len]
