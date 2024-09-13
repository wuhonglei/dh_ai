from torch import nn


# 定义编码器的类Encoder
class Encoder(nn.Module):
    # init函数是类的初始化函数，包括了5个参数
    # input_size是输入数据的维度，对应词汇表的大小
    # embed_size是词向量的维度
    # hidden_size是隐藏层神经元个数
    # n_layers是隐藏层数量
    # dropout是丢弃比率
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        # 定义Embedding层，大小为input_size*embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.drop = nn.Dropout(dropout)  # 定义drop层
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            dropout=dropout)

    # 在forward函数中，实现模型的前向传播
    # 函数传入输入序列src
    def forward(self, src):
        x = self.embed(src)  # 输入至Embedding层
        x = self.drop(x)  # 输入至drop层
        # 输入至lstm层，计算出隐藏状态hidden和记忆细胞cell
        output, (hidden, cell) = self.lstm(x)
        # 只需要修改forward函数的返回结果，将全部隐藏层的状态output返回
        return output, hidden, cell  # 返回hidden和cell


import torch.nn.functional as F


# 定义Attention类表示Attention层
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 定义两个线性层attn和v
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    # 函数传入当前时刻的解码器的输出ht和编码器的全部输出E
    def forward(self, ht, E):
        n = E.shape[0]  # E中包含的输出数量，也就是输入序列的长度
        ht_expanded = ht.repeat(n, 1, 1)  # 将ht复制n次
        # 这样后续才能将ht和每个ei组合，与编码器的n个输出对齐。

        ht_expanded = ht_expanded.transpose(0, 1)  # 调整维度的顺序
        E = E.transpose(0, 1)  # 调整维度的顺序
        # 将ht_expanded和E组合到一起，得到向量combine
        combine = torch.cat([ht_expanded, E], dim=2)

        # 将combine输入至线性层attn，使用tanh激活
        # 再输入至线性层v，得到能量值energy
        energy = self.v(torch.tanh(self.attn(combine)))
        # 将energy输入至softmax函数，计算出注意力权重
        attn_weights = F.softmax(energy, dim=1)
        # 返回调整维度顺序的注意力权重
        return attn_weights.transpose(1, 2)


# 定义解码器的类Decoder
class Decoder(nn.Module):
    # init函数是类的初始化函数，包括了5个参数
    # embed_size是词向量的维度
    # hidden_size是隐藏层神经元个数
    # output_size是输出数据的维度，对应翻译后的词汇表大小
    # n_layers是隐藏层数量
    # dropout是丢弃比率
    def __init__(self, embed_size, hidden_size, output_size, n_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = output_size  # 后续在seq2seq类中会使用到这个变量
        # 定义Embedding层，大小为output_size*embed_size，接收并处理翻译后的词语
        self.embed = nn.Embedding(output_size, embed_size)
        self.drop = nn.Dropout(dropout)  # 定义drop层
        # 定义LSTM层，修改LSTM层的尺寸
        self.lstm = nn.LSTM(input_size=embed_size + hidden_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            dropout=dropout)
        # 最后定义一个hidden_size*output_size大小的线性层fc，用来预测翻译词语
        # 修改线性层的尺寸
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention = Attention(hidden_size)  # 加入attention层

    # 在forward函数中，实现模型的前向传播
    # 函数传入前一个已经翻译好的单词input
    # 前一时刻的隐藏状态hidden和记忆细胞cell
    def forward(self, input, hidden, cell, encoder_out):
        # 将单个词的索引input扩展增加一个维度，变为序列长度*批量大小
        # 例如，将一个索引[1]，扩展成[[1]]
        x = input.unsqueeze(0)
        x = self.embed(x)  # 输入至Embedding层
        x = self.drop(x)  # 输入至drop层

        # 使用attention层，计算出attn_weights
        attn_weights = self.attention(hidden[-1], encoder_out)
        # 调整encoder_out的维度顺序
        encoder_out = encoder_out.transpose(0, 1)
        # 使用批量矩阵乘法torch.bmm，计算attn_weights和encoder_out的矩阵乘法
        context = torch.bmm(attn_weights, encoder_out)
        # 调整context的维度顺序
        context = context.transpose(0, 1)
        # x拼接，得到当前时刻lstm层的输入
        lstm_input = torch.cat([x, context], 2)

        # 输入至lstm层
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(0)  # 去掉最外层的序列长度维度
        context = context.squeeze(0)

        # 将lstm层的输出output与context拼接
        combine = torch.cat([output, context], 1)
        output = self.fc(combine)  # 使用fc层，计算出预测结果
        return output, hidden, cell  # 返回output、hidden和cell


import torch
import random


# seq2seq模型的定义
class Seq2Seq(nn.Module):
    # 将已声明的编码器Encoder、解码器Decoder
    # 还有当前使用的设备device传入
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        # 保存在seq2seq模型内
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # seq2seq模型有两种工作模式，分别是训练模式和预测模式
    # 两种工作模式，都会调用seq2seq模型的前向传播函数forward
    # 待翻译的文本序列src
    # 翻译后的结果序列trg
    # 变量teacher_forcing_ratio
    # 用于控制模型依赖真实数据和自身预测数据的比例
    def forward(self, src, trg, teacher_forcing_ratio):

        # 使用编码器encoder，将src转为语义向量C
        # 语义向量C即为encoder中的LSTM的返回结果
        # 隐藏状态hidden和记忆细胞cell
        # encoder_output是encoder(src)的计算出的所有隐藏状态
        encoder_output, hidden, cell = self.encoder(src)

        # seq2seq模型的期望输出result是trg
        # result的第1维和第2维和trg的第一维与第二维相同
        seq_len = trg.shape[0]  # 第1维是序列长度
        batch_size = trg.shape[1]  # 第2维是批量大小
        # 第3维是待预测词语的数量，也就是目标语言的词表大小
        # 保存在解码器decoder中的目标语言的词表大小
        vocab_size = self.decoder.vocab_size

        # 定义保存输出结果的张量result，result是一个三维的张量
        result = torch.zeros(seq_len, batch_size, vocab_size).to(self.device)

        # 在trg[0]中，会保存起始标记词<start>，对应的索引
        input = trg[0]  # 定义预测结果中的第1个词
        for i in range(1, seq_len):
            # 使用解码器decoder，一个接一个进行预测
            # 将encoder_output，传入到解码器decoder
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_output)
            # 将第i个预测结果output，保存在result[i]中
            result[i] = output
            # 赋值为预测结果中，概率最大的那个词
            input = output.argmax(dim=1)
            # 当随机数小于教师强制参数时
            if random.random() < teacher_forcing_ratio:
                input = trg[i]  # 直接将input赋值为正确的结果
        return result  # 返回result


# 初始化seq2seq模型参数的函数init_weights
# 通过初始化模型参数，可以加快模型的收敛，避免梯度消失或梯度爆炸
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

# 函数用于拆分已进行切词的句子
def split_token(s):
    # 将字符串s，以空格分隔出单词，将结果保存到列表中返回
    return [w for w in s.split(" ") if len(w) > 0]

from torch.utils.data import Dataset
# 定义TranslateDataset类，读取英译汉的训练数据
class TranslateDataset(Dataset):
    # init函数用于初始化，函数传入训练数据文件的路径path
    def __init__(self, path):
        file = open(path, 'r', encoding='utf-8') # 打开训练数据
        self.examples = list() # 保存英译汉的样本数据
        for line in file: # 循环读取数据中的每一行
            # 将每一行，根据\t字符，拆成源语言句子src和目标语言句子trg
            src, trg = line.strip().split('\t')
            # 使用split_token对src和trg拆分，在开头和结尾，补上标记词<sos>和<eos>
            # 分别表示句子的起始单词和结束单词
            src_tokens = ["<sos>"] + split_token(src) + ["<eos>"]
            trg_tokens = ["<sos>"] + split_token(trg) + ["<eos>"]
            # 将src_tokens和trg_tokens添加到examples
            self.examples.append((src_tokens, trg_tokens))

    def __len__(self):
        return len(self.examples) # 返回列表examples的长度

    def __getitem__(self, index):
        return self.examples[index] # 返回下标为index的数据

from torchtext.vocab import build_vocab_from_iterator
# 实现build_vocab函数，基于构建的数据集dataset，建立词汇表
# 词汇表包括两个，分别是源语言词汇表和目标语言词汇表
def build_vocab(dataset):
    # unk表示未知词，pad表示填充词，sos是起始标记，eos是结束标记
    special = ["<unk>", "<pad>", "<sos>", "<eos>"]
    src_iter = map(lambda x: x[0], dataset) # 源语言序列
    trg_iter = map(lambda x: x[1], dataset) # 目标语言序列
    # 建立源语言词汇表src_vocab和目标语言词汇表trg_vocab
    src_vocab = build_vocab_from_iterator(src_iter, specials = special)
    trg_vocab = build_vocab_from_iterator(trg_iter, specials = special)
    # 将unk对应的索引，设置为默认索引
    src_vocab.set_default_index(src_vocab["<unk>"])
    trg_vocab.set_default_index(trg_vocab["<unk>"])
    return src_vocab, trg_vocab # 返回两个词汇表

import torch
import nltk

def test_sample(model, src_vocab, trg_vocab, src, trg):
    trg_tokens = [token[0] for token in trg]
    trg_tokens = [trg_tokens[1:-1]]
    src_index = [src_vocab[token[0]] for token in src]
    src_tensor = torch.LongTensor(src_index).view(-1, 1).to(DEVICE)
    src_tensor = src_tensor.to(DEVICE)
    EOS_token = trg_vocab['<eos>']
    trg_index = [EOS_token for i in range(256)]
    trg_tensor = torch.LongTensor(trg_index).view(-1, 1).to(DEVICE)

    # 使用模型翻译src
    predict = model(src_tensor, trg_tensor, 0.0)
    predict = torch.argmax(predict.squeeze(1), dim=1).cpu()
    predict = predict[1:]
    trg_itos = trg_vocab.get_itos()
    predict_word = list()
    for id in predict:
        word = trg_itos[id]
        if word == '<eos>':
            break
        predict_word.append(word) #预测单词序列

    # 调用nltk中的sentence_bleu函数，对比预测翻译序列和目标序列
    # 计算BLEU分数，用于评估机器翻译质量
    bleu = nltk.translate.bleu_score.sentence_bleu(trg_tokens, predict_word)
    return predict_word, trg_tokens, bleu








import pickle
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 读入src_vocab和trg_vocab两个词汇表
    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("trg_vocab.pkl", "rb") as f:
        trg_vocab = pickle.load(f)

    # 定义模型的必要参数
    INPUT_DIM = len(src_vocab)
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    OUTPUT_DIM = len(trg_vocab)
    N_LAYERS = 2
    DROPOUT = 0.25
    # 定义编码器和解码器
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)
    # 将model转到DEVICE上
    enc = enc.to(DEVICE)
    dec = dec.to(DEVICE)

    # 定义seq2seq模型
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    # 使用load_state_dict，加载已经训练好的模型
    model.load_state_dict(torch.load('translate.model'))
    model.eval() # 模型设置为测试模式

    # 使用TranslateDataset加载测试数据集
    dataset = TranslateDataset("./data/small.data")
    # 定义dataloader读取dataset
    test_loader = DataLoader(dataset)
    print("total test num: %d"%(len(dataset)))

    # 遍历全部的测试样本
    for idx, (src, trg) in enumerate(test_loader):
        # 对于每个测试样本，调用函数test_sample测试效果
        predict, target, bleu_score = test_sample(model, src_vocab, trg_vocab, src, trg)
        # 打印测试结果
        print(f"{str(predict)} vs {str(target)} = {bleu_score:.2f}")
