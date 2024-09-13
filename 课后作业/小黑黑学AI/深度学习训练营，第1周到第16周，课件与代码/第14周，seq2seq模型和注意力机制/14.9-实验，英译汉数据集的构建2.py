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

# 使用TranslateDataset定义数据集dataset
dataset = TranslateDataset("data/small.data")
# 使用build_vocab，建立源语言和目标语言的词表
src_vocab, trg_vocab = build_vocab(dataset)

# 打印两个词表长度，和所包含的词语
print("src_vocab:", len(src_vocab))
print(src_vocab.get_itos())
print("trg_vocab:", len(trg_vocab))
print(trg_vocab.get_itos())






















