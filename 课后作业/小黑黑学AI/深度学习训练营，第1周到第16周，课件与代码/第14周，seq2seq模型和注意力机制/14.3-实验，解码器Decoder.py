from torch import nn

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

if __name__ == '__main__':
    EMB_DIM = 5 # 词向量维度
    HID_DIM = 6 # LSTM隐藏层中的神经元个数
    OUTPUT_DIM = 80  # 表示翻译后的词汇表的单词数量是80
    N_LAYERS = 2 # LSTM层中的隐藏层数量
    DROPOUT = 0.5 # 丢弃比率为0.5，即随机丢弃50%的神经元
    # 创建一个解码器模型model
    model = Decoder(EMB_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    # 打印模型model，可以看到Encoder模型的结构
    print(model)

    # 定义包含一个单词的索引input，这里的input相当于起始标记<start>
    input = torch.tensor([0])
    # hidden和cell可以理解为解码器的输出
    hidden = torch.zeros(N_LAYERS, 1, HID_DIM)
    cell = torch.zeros(N_LAYERS, 1, HID_DIM)

    # 使用model，可以计算出解码器的输出
    output, hidden, cell = model(input, hidden, cell)
    # 打印output的尺寸
    # 可以看到它是一个1*80大小的张量，相当于是翻译词表的大小
    print("Output shape:", output.shape)

    for i in range(5):
        # 在循环中，使用model，翻译并预测后面的5个单词
        output, hidden, cell = model(input, hidden, cell)
        print(f"{i} : {input}")
        input = output.argmax(dim = 1) # 更新单词的索引


