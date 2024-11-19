import torch
import torch.nn as nn
from dataset import split_token


def test_translate(model, src_vocab, trg_vocab, device):
    # 函数传入模型model和src_vocab与trg_vocab两个词表

    model.eval()
    sample = "<sos> I like math . <eos>"  # 定义一个测试样本
    src_tokens = split_token(sample)  # 分词结果
    src_index = [src_vocab[token] for token in src_tokens]  # 通过词表转为词语的索引
    src_tensor = torch.LongTensor(src_index).view(1, -1).to(device)  # 转为张量

    EOS_token = trg_vocab['<eos>']  # 获得目标序列的EOS
    # 填充一个包含256个EOS的句子，也就是最大的翻译结果长度是256
    trg_index = [EOS_token for i in range(256)]
    # 转为张量
    trg_tensor = torch.LongTensor(trg_index).view(1, -1).to(device)

    # 使用model预测翻译结果
    predict = model(src_tensor, trg_tensor)
    predict = torch.argmax(predict.squeeze(0), dim=1).cpu()
    predict = predict[1:]
    # 将预测结果转为词语序列
    trg_itos = trg_vocab.get_itos()
    predict_word = list()
    for id in predict:
        word = trg_itos[id.item()]
        if word == '<eos>':
            break
        predict_word.append(word)
    print("I like math . -> ", end="")
    print("".join(predict_word))  # 打印出来

# 函数传入模型model和src_vocab与trg_vocab两个词表


def test_samples(model, src_tokens, target_tokens, src_vocab, trg_vocab):
    model.eval()
    device = model.device

    target_index = [trg_vocab[token]
                    for token in target_tokens][1:-1]  # 通过词表转为词语的索引
    target_word = [trg_vocab[id] for id in target_index]

    src_index = [src_vocab[token] for token in src_tokens]  # 通过词表转为词语的索引
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
    predict_word = list()
    for id in predict:
        word = trg_vocab[id.item()]
        if word == '<eos>':
            break
        predict_word.append(word)

    bleu = sentence_bleu([target_word], predict_word)
    return predict_word, target_word, bleu
