import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))  # 可学习的参数向量

    def forward(self, hidden, encoder_outputs):
        # hidden: 当前解码器的隐藏状态 [batch_size, hid_dim]
        # encoder_outputs: 编码器的所有隐藏状态 [batch_size, src_len, hid_dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # 复制 hidden 以匹配 src_len 维度
        hidden = hidden.unsqueeze(1).repeat(
            1, src_len, 1)  # [batch_size, src_len, hid_dim]

        # 计算注意力权重
        # [batch_size, src_len, hid_dim]
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)  # [batch_size, hid_dim, src_len]

        # 使用 v 向量对 energy 进行线性变换
        v = self.v.repeat(batch_size, 1).unsqueeze(
            1)  # [batch_size, 1, hid_dim]
        attention = torch.bmm(v, energy).squeeze(
            1)  # [batch_size, src_len]

        # 通过 softmax 归一化
        return F.softmax(attention, dim=1)  # [batch_size, src_len]
