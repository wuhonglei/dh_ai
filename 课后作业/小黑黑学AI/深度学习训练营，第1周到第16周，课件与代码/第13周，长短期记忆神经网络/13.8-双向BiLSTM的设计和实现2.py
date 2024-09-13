import string
# 设置一个全局变量all_letters
# 保存了训练数据中全部可能出现的字符
# 包括英文的大小写，加上空格、点、逗号、分号、引号等标点符号
all_letters = string.ascii_letters + " .,;'"

import unicodedata
# 实现一个unicode转Asc2码的函数，这个函数的主要作用是，将拉丁字符转为英文字符
# 在未来在训练的时候，我们只关注英文单词中的大小写字符
# 一些语言中的特殊字符，会直接转为英文的大小
def unicode_to_asc2(name):
    result = ""  # 保存转换后的结果
    # 对输入的name进行标准化
    name_asc2 = unicodedata.normalize('NFD', name)
    # 遍历标准化后的字符串中的字符
    for c in name_asc2:
        # 如果字符c不是词素记号，例如不是重音符号，并且c还是英文字符
        if unicodedata.category(c) != 'Mn' and c in all_letters:
            result += c #将c添加到结果中
    return result #返回结果


import os
# 该函数传入待读入的文件路径，返回一个列表
# 列表中保存了该文件中的全部名字单词
def read_names_file(filename):
    names = list() # 保存名字单词的结果
    # 将文件打开，并将文件中的数据读入到变量lines中，数据会按行进行读入
    lines = open(filename, encoding='utf-8').readlines()
    for line in lines: # 遍历文件中的每一行
        line = line.strip() # 去掉该行的首尾空格
        # 调用unicode_to_asc2，对单词进行转换
        names.append(unicode_to_asc2(line)) #将结果添加到names
    return names #函数返回names

from torch.utils.data import Dataset
# 设置NamesDataset继承Dataset，用于读取名字训练数据
class NamesDataset(Dataset):
    # init函数用于初始化
    # init函数用于初始化，函数传入数据的路径data_dir
    def __init__(self, data_dir):
        # 定义name和label，保存名字数据和该数据的类别
        self.name = list()
        self.label = list()
        self.label_name = list() #label_name保存全部可能的标签
        # 获取data_dir中的全部文件，保存在files中
        files = os.listdir(data_dir)

        for file in files:  # 遍历files
            # 将目录与文件名组合为文件路径，保存在path中
            path = os.path.join(data_dir, file)
            # 读取path对应文件中的名字数据
            names = read_names_file(path)
            # 获取到保存数据的文件名，文件名即为对应的类别标签
            label = os.path.splitext(os.path.basename(path))[0]
            self.label_name.append(label) #将类别标签保存到label_name中
            for name in names:
                # 将每个名字和标签组合到一起，成为一个数据
                # 添加到label和name的列表中
                self.label.append(label)
                self.name.append(name)
        self.length = len(self.label) #保存数据的个数

    def __len__(self):
        return self.length #返回数据集中的样本数量

    # 函数getitem传入索引index
    def __getitem__(self, index):
        # 返回与该索引对应的数据name[index]和标签label[index]
        return self.name[index], self.label[index]

    # 获得类别标签的数量
    def get_labels_num(self):
        return len(self.label_name)

    # 根据类别名称得到对应的编号
    def get_label_index(self, name):
        return self.label_name.index(name)

    # 根据类别编号得到类别的名字
    def get_label_name(self, index):
        return self.label_name[index]

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
# 实现一个BiLSTM模型，用来训练名字分类模型
class BiLSTMModel(nn.Module):
    # init函数传入输入层、隐藏层和输出层的神经元数据量
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        # 定义正向LSTM层lstm_forward
        self.lstm_forward = nn.LSTM(input_size, hidden_size)
        # 定义逆向LSTM层lstm_backward
        self.lstm_backward = nn.LSTM(input_size, hidden_size)
        # 定义隐藏层到输出层的线性层h2o
        # 接收正向和逆向合并后的数据
        # 因此该层的输入数据长度是hidden_size * 2
        self.h2o = nn.Linear(hidden_size * 2, output_size)

    # 实现前向传播函数forward，函数直接传入输入向量input
    def forward(self, input):
        # 计算正向LSTM的结果
        # 将最后一个时刻的隐藏层输出保存在ht_forward中
        _, (ht_forward, _) = self.lstm_forward(input)
        # 将输入数据input，通过flip函数
        # 在时间维度上进行翻转，得到逆序序列reversed
        reversed = torch.flip(input, [0])
        # 使用逆向LSTM计算结果，结果保存在ht_backward中
        _, (ht_backward, _) = self.lstm_backward(reversed)
        # 将正向结果和逆向结果进行合并，得到拼接结果combined
        combined = torch.cat((ht_forward[-1], ht_backward[-1]), dim = 1)
        output = self.h2o(combined) # 使用线性层h2o计算输出结果
        return output #将output返回

# 基于混淆矩阵，计算准确率和召回率
# 函数传入混淆矩阵confusion
def calculate_precision_recall(confusion):
    n = len(confusion) #类别的个数
    precision = [0] * n #保存准确率的数组
    recall = [0] * n #保存召回率的数组
    for i in range(n): #遍历n个类别
        tp = confusion[i][i] #对角线元素
        fp = 0
        fn = 0
        # 循环计算出fp和fn
        for j in range(n):
            if j != i:
                fp += confusion[j][i]
                fn += confusion[i][j]
        # 使用tp、fp、fn计算precision[i]和recall[i]
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
    return precision, recall #返回precision和recall

from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 使用NamesDataset读取测试数据集
    # 这里直接使用训练时所用的数据集
    dataset = NamesDataset("./data/names/")
    print("dataset length = %d" % (len(dataset)))
    print("labels num = %d" % (dataset.get_labels_num()))
    print("labels:")
    print(dataset.label_name)

    # 使用DataLoader读取test_data
    # 此时不需要设置任何参数，这样会一个一个的读取数据
    test_loader = DataLoader(dataset)

    n_letters = len(all_letters)
    n_hidden = 128
    n_classes = dataset.get_labels_num()
    # 创建一个BiLSTMModel模型
    model = BiLSTMModel(n_letters, n_hidden, n_classes)
    # 读取已经训练好的模型文件names.classify
    model.load_state_dict(torch.load('names.classify.pytorch.bilstm'))

    right = 0  # 设置right变量，保存预测正确的样本数量
    all = 0  # all保存全部的样本数量
    # 保存混淆矩阵
    confusion = torch.zeros(n_classes, n_classes)

    # 遍历test_loader中的数据
    for (data, label) in test_loader:

        input = name_to_tensor(data[0])

        # 使用模型预测样本的结果
        output = model(input)

        output_idx = output.argmax(1).item() #样本的预测索引
        # 计算样本标记索引
        label_idx = dataset.get_label_index(label[0])
        confusion[label_idx][output_idx] += 1 #给混淆矩阵赋值

        predict = dataset.get_label_name(output_idx) #预测结果
        if predict == label[0]: #检查pred和y是否相同
            right += 1  # 如果相同，那么right加1
        all += 1  # 每次循环，all变量加1，记录已计算的数据个数

    acc = right * 1.0 / all # 循环结束后，计算模型的正确率
    print("test accuracy = %d / %d = %.3lf" % (right, all, acc))

    # 计算全部类别的准确率和召回率，并将结果打印出来
    precision, recall = calculate_precision_recall(confusion)
    for i in range(n_classes):
        print("%s precision = %.2lf recall = %.2lf"%
              (dataset.label_name[i], precision[i], recall[i]))

