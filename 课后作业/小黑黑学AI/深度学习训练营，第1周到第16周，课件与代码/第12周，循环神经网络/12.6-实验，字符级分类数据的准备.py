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


if __name__ == '__main__':
    s = 'Ślusàrski' #转换拉丁文s
    print("%s -> %s"%(s, unicode_to_asc2(s)))

    # 读取训练数据集
    dataset = NamesDataset("./data/names/")

    # 打印读取的数据数量、标签数量和标签名
    print("dataset length = %d" % (len(dataset)))
    print("labels num = %d" % (dataset.get_labels_num()))
    print("labels:")
    print(dataset.label_name)



