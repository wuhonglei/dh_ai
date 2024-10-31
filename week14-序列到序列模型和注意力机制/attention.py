import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h_t, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = h_t.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        combined = torch.cat((h, encoder_outputs), dim=2)
        energy = self.v(torch.tanh(self.attn(combined)))
        attention_weights = torch.softmax(energy, dim=1)
        return attention_weights.transpose(1, 2)


if __name__ == '__main__':
    batch_size = 2  # 定义 batch_size
    hidden_size = 3  # 定义隐藏层大小
    attention = Attention(hidden_size)  # 定义 attention
    # 随机生成一个 ht, 维度为 (1, batch_size, 3), 表示 序列长度为 1, batch_size 为 2, 隐藏层大小为 hidden_size
    h_t = torch.randn(1, batch_size, hidden_size)
    # 定义编码器的输出, 维度为 (5, batch_size, 3), 表示 序列长度为 5, batch_size 为 2, 隐藏层大小为 hidden_size
    encoder_outputs = torch.randn(5, batch_size, hidden_size)
    attention_weights = attention(h_t, encoder_outputs)
    print(attention_weights.size())
