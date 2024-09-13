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

if __name__ == '__main__':
    # 打印所有可能的字符集合
    print("all_letters = %s"%(all_letters))
    name = 'Albert' #设置name等于Albert
    tensor = name_to_tensor(name) #将字符串name，转为张量
    print("tensor size: ", tensor.size()) #打印张量的维度
    # 打印名字name中的每个字符name[i]和它对应的张量tensor[i]
    for i in range(len(tensor)):
        print(f"{name[i]} : {tensor[i]}")


