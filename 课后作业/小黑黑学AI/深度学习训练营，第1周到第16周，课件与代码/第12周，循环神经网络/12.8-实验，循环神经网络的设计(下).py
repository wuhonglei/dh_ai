import string
# 设置一个全局变量all_letters
# 保存了训练数据中全部可能出现的字符
# 包括英文的大小写，加上空格、点、逗号、分号、引号等标点符号
all_letters = string.ascii_letters + " .,;'"

import torch
# 将传入的名字单词name转为张量
def name_to_tensor(name):
    # 定义序列长度×批量大小×字符维度的张量
    tensor = torch.zeros(len(name), 1, len(all_letters))
    for i, letter in enumerate(name): # 遍历word中的字符
        # 计算当前遍历的字符letter在所有字符中的索引位置index
        index = all_letters.find(letter)
        # 将该字符对应的张量tensor[i][0]的第index位置设置为1
        tensor[i][0][index] = 1 # 完成了第i个字符的one-hot
    return tensor # 函数返回tensor

import torch.nn as nn
# 定义类RNNModel，它继承了torch.nn中的Module模块
class RNNModel(nn.Module):
    # init函数传入输入层、隐藏层和输出层的神经元数据量
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # 隐藏层神经元的个数，会在创建初始隐藏层输出向量使用
        self.hidden_size = hidden_size #保存隐藏层神经元个数
        # 定义输入层到隐藏层的线性层i2h
        # 它是一个(input_size + hidden_size)×hidden_size大小的线性层
        # 例如，在我们的实验中，它的大小是(57+128)*128
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 定义隐藏层到输出层的线性层h2o
        # 它的大小是hidden_size×output_size
        # 在我们的实验中，它的大小是128×12
        self.h2o = nn.Linear(hidden_size, output_size)
        

    # 前向传播函数forward，函数输入两个张量
    # input为每个时刻的输入数据，hidden为上一个时刻的隐藏层输出数据
    def forward(self, input, hidden):
        # 将张量input和hidden连接在一起
        combined = torch.cat((input, hidden), 1)
        # 使用线性层i2h，计算当前时刻的隐藏层输出hidden
        hidden = self.i2h(combined)
        # 使用线性层h2o，计算当前时刻的输出层输出output
        output = self.h2o(hidden)
        return hidden, output #返回hidden和output

    # 为了初始化第一个隐藏层输出，设置函数init_hidden
    # 该函数用于声明一个1×hidden_size大小空张量
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__ == '__main__':
    word = 'Albert' #定义名字单词
    tensor = name_to_tensor(word) #将word转为张量
    print("input size: ", tensor.size()) #输出tensor的尺寸

    model = RNNModel(len(all_letters), 128, 12) #定义循环神经网络
    # 使用model，初始化一个空的隐藏层输出hidden
    hidden = model.init_hidden()
    letters_num = tensor.size()[0] #计算待输入字符的数量

    # 使用循环计算输入序列tensor的结果
    for i in range(letters_num):
        # 将每个时刻的字符tensor[i]
        # 与上一时刻的隐藏层输出hidden输入至模型model
        # 计算出这一时刻的隐藏层输出hidden和输出层输出output
        hidden, output = model(tensor[i], hidden)
        print("i = %d word = %s"%(i, word[i])) #打印当前的字符
        print(f"\tinput: {tensor[i].shape}") #输入张量尺寸
        print(f"\thidden: {hidden.shape}") #隐藏层输出的张量尺寸
        print(f"\toutput: {output.shape}") #输出层输出的张量尺寸



