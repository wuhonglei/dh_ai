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
        self.drop = nn.Dropout(dropout) # 定义drop层
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = n_layers,
                            dropout = dropout)

    # 在forward函数中，实现模型的前向传播
    # 函数传入输入序列src
    def forward(self, src):
        x = self.embed(src) # 输入至Embedding层
        x = self.drop(x) # 输入至drop层
        # 输入至lstm层，计算出隐藏状态hidden和记忆细胞cell
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell # 返回hidden和cell


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
        self.vocab_size = output_size #后续在seq2seq类中会使用到这个变量
        # 定义Embedding层，大小为output_size*embed_size，接收并处理翻译后的词语
        self.embed = nn.Embedding(output_size, embed_size)
        self.drop = nn.Dropout(dropout) # 定义drop层
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = n_layers,
                            dropout = dropout)
        # 最后定义一个hidden_size*output_size大小的线性层fc，用来预测翻译词语
        self.fc = nn.Linear(hidden_size, output_size)

    # 在forward函数中，实现模型的前向传播
    # 函数传入前一个已经翻译好的单词input
    # 前一时刻的隐藏状态hidden和记忆细胞cell
    def forward(self, input, hidden, cell):
        # 将单个词的索引input扩展增加一个维度，变为序列长度*批量大小
        # 例如，将一个索引[1]，扩展成[[1]]
        x = input.unsqueeze(0)
        x = self.embed(x) # 输入至Embedding层
        x = self.drop(x) # 输入至drop层
        # 输入至lstm层
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = output.squeeze(0) # 去掉最外层的序列长度维度
        output = self.fc(output) # 使用fc层，计算出预测结果
        return output, hidden, cell # 返回output、hidden和cell

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
        hidden, cell = self.encoder(src)

        # seq2seq模型的期望输出result是trg
        # result的第1维和第2维和trg的第一维与第二维相同
        seq_len = trg.shape[0] # 第1维是序列长度
        batch_size = trg.shape[1] # 第2维是批量大小
        # 第3维是待预测词语的数量，也就是目标语言的词表大小
        # 保存在解码器decoder中的目标语言的词表大小
        vocab_size = self.decoder.vocab_size
        
        # 定义保存输出结果的张量result，result是一个三维的张量
        result = torch.zeros(seq_len, batch_size, vocab_size).to(self.device)

        # 在trg[0]中，会保存起始标记词<start>，对应的索引
        input = trg[0] # 定义预测结果中的第1个词
        for i in range(1, seq_len):
            # 使用解码器decoder，一个接一个进行预测
            output, hidden, cell = self.decoder(input, hidden, cell)
            # 将第i个预测结果output，保存在result[i]中
            result[i] = output
            # 赋值为预测结果中，概率最大的那个词
            input = output.argmax(dim = 1)
            # 当随机数小于教师强制参数时
            if random.random() < teacher_forcing_ratio:
                input = trg[i] # 直接将input赋值为正确的结果
        return result # 返回result


# 初始化seq2seq模型参数的函数init_weights
# 通过初始化模型参数，可以加快模型的收敛，避免梯度消失或梯度爆炸
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

if __name__ == '__main__':
    INPUT_DIM = 100  # 输入词汇表大小
    OUTPUT_DIM = 80  # 输出词汇表大小
    EMB_DIM = 8  # 输出词汇表大小
    HID_DIM = 16  # 隐藏层维度
    N_LAYERS = 2 # LSTM中的隐藏层数量
    DROPOUT = 0.5 # 丢弃比率
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义编码器
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    # 定义解码器
    decoder = Decoder(EMB_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    # 定义seq2seq模型
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.apply(init_weights) # 对模型进行初始化

    # 创建7*3大小的源序列src
    src = torch.randint(0, INPUT_DIM, (7, 3)).to(DEVICE)
    # 创建5*3大小目标序列trg
    trg = torch.randint(0, OUTPUT_DIM, (5, 3)).to(DEVICE)
    TEACHER_FORCING_RATIO = 0.5
    # 调用model，计算出结果output
    output = model(src, trg, TEACHER_FORCING_RATIO)
    print("output shape:", output.shape)

