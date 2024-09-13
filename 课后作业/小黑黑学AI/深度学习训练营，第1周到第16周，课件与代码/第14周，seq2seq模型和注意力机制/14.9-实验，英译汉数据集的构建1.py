# 函数用于拆分已进行切词的句子
def split_token(s):
    # 将字符串s，以空格分隔出单词，将结果保存到列表中返回
    return [w for w in s.split(" ") if len(w) > 0]

s = "I am a student ."
# 将句子s切分为5个单词
print(split_token(s))

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

# 在small.data文件中，保存了5个测试样本
# 使用TranslateDataset读取数据集
dataset = TranslateDataset("data/small.data")
print(len(dataset)) # 打印数据集的长度
print(dataset[0][0]) # 打印第1个样本中的英文序列
print(dataset[0][1]) # 打印第1个样本中的中文序列

