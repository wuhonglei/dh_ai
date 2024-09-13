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

    # 在forward函数中，实现模型的前向传播，函数传入输入序列src
    def forward(self, src):
        x = self.embed(src) # 输入至Embedding层
        x = self.drop(x) # 输入至drop层
        # 输入至lstm层，计算出隐藏状态hidden和记忆细胞cell
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell # 返回hidden和cell

# 实现一个打印前向传播过程的函数
# 函数传入模型model和输入数据src
def print_forward(model, src):
    # 按照前向传播中的计算顺序，计算中间的结果并打印结果的尺寸
    print("src: ", src.shape)
    x = model.embed(src)
    print("embed: ", x.shape)
    x = model.drop(x)
    print("dropout: ", x.shape)
    _, (hidden, cell) = model.lstm(x)
    print("hidden: ", hidden.shape)
    print("cell: ", cell.shape)

import torch

if __name__ == '__main__':
    INPUT_DIM = 100  # 表示词汇表的单词数量是100
    EMB_DIM = 5  # 词向量维度
    HID_DIM = 6  # LSTM隐藏层中的神经元个数
    N_LAYERS = 2  # LSTM层中的隐藏层数量
    DROPOUT = 0.5  # 丢弃比率为0.5，即随机丢弃50%的神经元
    # 创建一个编码器模型
    model = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    # 打印模型model，可以看到Encoder模型的结构
    print(model)

    seq_length = 7 # 输入序列的长度
    batch_size = 3 # 数据个数
    # 定义7*3的张量src，代表3个长度是7的序列
    src = torch.randint(0, INPUT_DIM, (seq_length, batch_size))
    hidden, cell = model(src) # 计算出编码器的输出hidden和cell
    print_forward(model, src) # 打印model处理src的过程












