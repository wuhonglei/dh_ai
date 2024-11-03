import torch
import torch.nn as nn

from dataset import split_token


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


# 函数传入模型model和src_vocab与trg_vocab两个词表
def test_translate(model, src_vocab, trg_vocab):
    model.eval()
    device = model.device
    sample = "<sos> I like math . <eos>"  # 定义一个测试样本
    src_tokens = split_token(sample)  # 分词结果
    src_index = [src_vocab.get(token, src_vocab['<unk>'])
                 for token in src_tokens]  # 通过词表转为词语的索引
    src_tensor = torch.LongTensor(src_index).view(1, -1).to(device)  # 转为张量

    EOS_token = trg_vocab['<eos>']  # 获得目标序列的EOS
    # 填充一个包含256个EOS的句子，也就是最大的翻译结果长度是256
    trg_index = [EOS_token for i in range(256)]
    # 转为张量
    trg_tensor = torch.LongTensor(trg_index).view(1, -1).to(device)

    # 使用model预测翻译结果
    predict = model(src_tensor, trg_tensor, 0.0)
    predict = torch.argmax(predict.squeeze(0), dim=1).cpu()
    predict = predict[1:]
    # 将预测结果转为词语序列
    trg_itos = trg_vocab.get_itos()
    predict_word = list()
    for id in predict:
        word = trg_itos[id]
        if word == '<eos>':
            break
        predict_word.append(word)
    print("I like math . -> ", end="")
    print("".join(predict_word))  # 打印出来
