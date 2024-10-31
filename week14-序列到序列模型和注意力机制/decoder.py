import torch
import torch.nn as nn

from torchinfo import summary
from attention import Attention


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(output_size, embed_size)
        # 拼接 context 和 embedding 后的输入维度为 embed_size + hidden_size
        self.rnn = nn.LSTM(
            embed_size + hidden_size, hidden_size, num_layers, dropout=p, batch_first=True)
        # 拼接 context 和 hidden 后的输入维度为 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, input, hidden, cell, encoder_output):
        # input: 当前解码器的输入词 [batch_size]
        # hidden: 上一个时间步的隐藏状态 [num_layers, batch_size, hid_dim]
        # cell: 上一个时间步的细胞状态 [batch_size, hid_dim]
        # encoder_outputs: 编码器的所有隐藏状态 [batch_size, src_len, hid_dim]

        input = input.unsqueeze(1)  # [batch_size] => [batch_size, 1]
        # [batch_size, 1, embed_size]
        embedding = self.dropout(self.embedding(input))

        # 计算注意力权重
        a = self.attention(hidden[-1], encoder_output)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(a, encoder_output)  # [batch_size, 1, hid_dim]

        # 解码器的输入为上下文向量和嵌入向量的拼接
        # [batch_size, 1, embed_size + hid_dim]
        rnn_input = torch.concat((embedding, context), dim=2)

        # 通过 RNN 计算下一个隐藏状态
        # output: [batch_size, 1, hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)  # [batch_size, hid_dim]
        context = context.squeeze(1)  # [batch_size, hid_dim]

        # 最终输出预测结果
        output = self.fc(torch.concat((output, context), dim=1))
        return output, hidden, cell
