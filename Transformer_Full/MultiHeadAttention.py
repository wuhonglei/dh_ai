import torch.nn as nn
from util import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义线性映射层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并拆分为多头
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)

        # 计算注意力
        attn_output, attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 拼接多头
        attn_output = attn_output.transpose(
            1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 通过最终的线性层
        output = self.W_O(attn_output)
        output = self.dropout(output)
        return output
