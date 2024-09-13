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
# 定义类RNNModel，它继承了torch.nn中的Module模块
class RNNModel(nn.Module):
    # init函数传入输入层、隐藏层和输出层的神经元数据量
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # 直接使用Pytorch中的RNN模型
        self.rnn = nn.RNN(input_size, hidden_size)
        # 定义隐藏层到输出层的线性层h2o
        self.h2o = nn.Linear(hidden_size, output_size)

    # 实现前向传播函数forward，函数直接传入输入向量input
    def forward(self, input):
        # 使用rnn层计算最终的隐藏层输出ht
        _, ht = self.rnn(input)
        # 将最后一个隐藏层的t时刻的输出ht[-1]，输入至线性层h2o
        output = self.h2o(ht[-1])
        return output #返回output


from torch.utils.data import DataLoader
from torch import optim

if __name__ == '__main__':
    # 使用已经实现的NamesDataset模块，读取名字训练数据
    dataset = NamesDataset("./data/names/")
    # 打印读取的数据数量、标签数量和标签名
    print("dataset length = %d" % (len(dataset)))
    print("labels num = %d" % (dataset.get_labels_num()))
    print("labels:")
    print(dataset.label_name)

    # 使用DataLoader，定义数据加载器train_load
    # 参数dataset是训练集，batch_size=1代表每个小批量数据的大小是1
    # shuffle = True表示每一轮训练，都会随机打乱数据的顺序
    # 因此，这里使用了随机梯度下降算法进行训练
    train_load = DataLoader(dataset,
                            batch_size = 1,
                            shuffle = True)

    n_letters = len(all_letters) # 输入神经元个数为字符种类的数量
    n_hidden = 128 # 隐藏层的神经元数量
    n_classes = dataset.get_labels_num() # 分类的类别数量
    model = RNNModel(n_letters, n_hidden, n_classes) # 模型对象
    optimizer = optim.Adam(model.parameters())  # 创建Adam优化器
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

    # 进入模型的循环迭代
    for epoch in range(10):  # 外层循环，代表了整个训练数据集的遍历次数
        loss_sum = 0 # 用来打印调试信息使用

        # 内层循环代表了，在一个epoch中
        # 以随机梯度下降的方式，使用train_load对于数据进行遍历
        # batch_idx表示当前遍历的批次
        # (data, label) 表示这个批次的训练数据和标记
        for batch_idx, (data, label) in enumerate(train_load):
            tensor = name_to_tensor(data[0]) #保存数据data[0]的张量

            # 前向传播:
            output = model(tensor)
            # 前向传播结束

            # 获取当前数据标签label[0]对应的类别索引
            label_idx = dataset.get_label_index(label[0])
            # 将label_idx转为张量，保存在label_tensor中
            label_tensor = torch.tensor([label_idx],
                                        dtype=torch.long)

            # 反向传播:
            # 计算预测值output与真实值label之间的损失loss
            loss = criterion(output, label_tensor)
            loss.backward()  # 计算损失函数关于模型参数的梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 将梯度清零，以便于下一次迭代
            # 反向传播结束

            loss_sum += loss.item() # 累加当前样本的损失
            # 每训练1000个数据，打印一次损失
            if (batch_idx + 1) % 1000 == 0:
                # 获取输出output中最大值的索引值
                output_idx = output.argmax(1).item()
                # 获取该索引对应的预测结果
                predict = dataset.get_label_name(output_idx)
                # 比较预测结果predict和标记结果label[0]是否相同
                if predict == label[0]:
                    correct = '✓' #结果保存
                else:
                    correct = '✗'
                # 打印调试信息
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx + 1}/{len(train_load)} "
                      f"| Loss: {loss_sum:.4f}"
                      f"| {correct} {data[0]} {label[0]} {predict}")
                loss_sum = 0

    # 将训练好的模型保存为文件，文件名为names.classify
    torch.save(model.state_dict(), 'names.classify.pytorch.rnn')

