import torch
import torch.nn as nn
from dataset import split_token
from utils.model import create_subsequent_mask, create_pad_mask
from nltk.translate.bleu_score import sentence_bleu
from dataset import Vocab
from Transformer import Transformer


def translate_sentence(model: Transformer, src_sentence: str, src_vocab: Vocab, tgt_vocab: Vocab, device, max_len=50):
    """
    使用训练好的模型翻译源语言句子
    :param model: 训练好的 Transformer 模型
    :param src_sentence: 源语言句子（字符串）
    :param src_vocab: 源语言词汇表（词到索引的映射）
    :param tgt_vocab: 目标语言词汇表（索引到词的映射）
    :param device: 设备（'cpu' 或 'cuda'）
    :param max_len: 生成序列的最大长度
    :return: 生成的目标语言句子
    """
    model.eval()

    # 将源句子分词并转换为索引
    src_tokens = [src_vocab[token] for token in src_sentence.strip().split()]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(
        0).to(device)  # (1, src_seq_len)

    # 创建源句子的掩码 [1, 1, 1, src_seq_len]
    src_mask = create_pad_mask(src_tensor, src_vocab['<pad>']).to(device)

    # 获取编码器的输出
    memory = model.encoder(src_tensor, src_mask)

    # 初始化解码器输入（以 <sos> 开始）
    tgt_indices = [tgt_vocab['<sos>']]
    for i in range(max_len):
        # 初始输入 [1, tgt_seq_len]
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(
            0).to(device)
        tgt_pad_mask = create_pad_mask(
            tgt_tensor, tgt_vocab['<pad>']).to(device)
        tgt_sub_mask = create_subsequent_mask(tgt_tensor.size(1)).to(device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(
            0)  # [1, tgt_seq_len, tgt_seq_len]
        output = model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
        output = output.squeeze(0)  # (tgt_seq_len, tgt_vocab_size)
        next_token_logits = output[-1, :]  # 取最后一个时间步(最后一个单词)的输出

        # 选取概率最大的词作为下一个词
        next_token = next_token_logits.argmax(-1).item()
        tgt_indices.append(next_token)

        # 如果生成了 <eos>，则停止生成
        if next_token == tgt_vocab['<eos>']:
            break

    # 将索引转换为词
    tgt_tokens: list[str] = [tgt_vocab[idx]
                             for idx in tgt_indices[1:]]  # 去掉 <sos> # type: ignore

    return ' '.join(tgt_tokens)


def test_samples(model, src_tokens, target_tokens, src_vocab, trg_vocab):
    # 函数传入模型model和src_vocab与trg_vocab两个词表
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
