from torch.utils.data import DataLoader
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

# 第一步是数据集的构造
# 无论是计算机视觉任务还是NLP任务，都要继承Dataset，
# 构造一个用于读取数据的数据集


class NewsDataset(Dataset):
    # init函数传入data对象
    # data对象中保存了原始数据，包括标签label和新闻描述text
    # data[0] = (3, "Wall St. Bears Claw Back Into the...")
    def __init__(self, data):
        self.examples = list()
        tokenizer = get_tokenizer("basic_english")
        for i, (label, text) in enumerate(data):  # 使用循环，遍历data
            # 将文本转为小写，并使用tokenizer进行分词
            tokenized_text = [token.lower() for token in tokenizer(text)]
            # 将处理好的文本和标签，保存到examples中
            self.examples.append((tokenized_text, label))

    def __len__(self):
        return len(self.examples)  # 返回examples的长度

    def __getitem__(self, index):
        return self.examples[index]  # 返回第index个数据


# 实现build_vocab函数，基于数据集dataset，建立词汇表
# 词汇表包括两个，分别是文本词汇表和标注词汇表


def build_vocab(dataset):
    # unk表示未知词，pad表示填充词
    special = ["<unk>", "<pad>"]

    text_iter = map(lambda x: x[0], dataset)  # 文本序列
    # 建立文本词汇表text_vocab和目标语言词汇表pos_vocab
    # 将min_freq设置为2，也就是至少出现两次的单词，才会添加到词表中
    text_vocab = build_vocab_from_iterator(
        text_iter, min_freq=2, specials=special)
    # 将unk对应的索引，设置为默认索引
    text_vocab.set_default_index(text_vocab["<unk>"])
    return text_vocab


# 整个collate_batch函数是批量读取文本数据时的回调函数


def collate_batch(batch, text_vocab):
    text_list = list()
    labels = list()
    # 每次读取一组数据
    for text, label in batch:
        text_tokens = [text_vocab[token] for token in text]
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_list.append(text_tensor)
        labels.append(torch.tensor(label - 1, dtype=torch.long))

    padding_idx = text_vocab["<pad>"]
    # 将batch填充为相同长度文本
    text_padded = pad_sequence(
        text_list, batch_first=True, padding_value=padding_idx)
    # print("text_padded:", text_padded.shape)
    labels_tensor = torch.stack(labels)

    # 返回文本和标签的张量形式，用于后续的模型训练
    return text_padded, labels_tensor


if __name__ == '__main__':
    train_data, _ = AG_NEWS()

    train_data = list(train_data)
    print("train_data length: %d" % (len(train_data)))

    dataset = NewsDataset(train_data)  # 创建数据集dataset
    text_vocab = build_vocab(dataset)  # 创建词表

    def collate(batch): return collate_batch(batch, text_vocab)
    # 接着定义dataloader读取dataset
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            collate_fn=collate)

    for batch_idx, (text, label) in enumerate(dataloader):  # 遍历dataloader
        if batch_idx >= 3:
            break
        # 打印每个小批次
        print("batch_idx:", batch_idx)
        print("text:")
        print(text)
        print("label:")
        print(label)
