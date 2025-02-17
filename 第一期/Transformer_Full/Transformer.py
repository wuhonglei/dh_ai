import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from utils.model import create_pad_mask, create_subsequent_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model,
                               num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model,
                               num_layers, num_heads, d_ff, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器输出 [batch_size, src_seq_len, d_model]
        memory = self.encoder(src, src_mask)
        # 解码器输出 [batch_size, tgt_seq_len, tgt_vocab_size]
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        return output


if __name__ == "__main__":
    # 假设源语言和目标语言的词汇表大小
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    pad_idx = 0  # 假设填充符的索引为 0

    # 定义模型参数
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # 创建模型
    model = Transformer(src_vocab_size, tgt_vocab_size,
                        d_model, num_layers, num_heads, d_ff, dropout)

    # 假设输入序列和目标序列
    # [batch_size, src_seq_len]
    src = torch.randint(1, src_vocab_size, (32, 20))
    # [batch_size, tgt_seq_len]
    tgt = torch.randint(1, tgt_vocab_size, (32, 20))

    # 创建掩码
    src_mask = create_pad_mask(src, pad_idx)  # [batch_size, 1, 1, src_seq_len]
    # [batch_size, 1, 1, tgt_seq_len]
    tgt_pad_mask = create_pad_mask(tgt, pad_idx)
    tgt_sub_mask = create_subsequent_mask(tgt.size(1)).to(
        tgt.device)  # [tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)  # 合并掩码

    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)
    print(output.shape)  # 应输出 [batch_size, tgt_seq_len, tgt_vocab_size]
