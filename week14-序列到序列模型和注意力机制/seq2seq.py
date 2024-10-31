import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_force_ratio=0.5):
        """
        src: [batch_size, seq_len]
        trg: [batch_size, seq_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size).to(self.device)

        encoder_output, hidden, cell = self.encoder(src)
        input = trg[:, 0]  # 取 <sos> 标签作为初始解码器输入, [batch_size]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(
                input, hidden, cell, encoder_output)
            outputs[:, t] = output
            teacher_force = torch.rand(
                1).item() < teacher_force_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
