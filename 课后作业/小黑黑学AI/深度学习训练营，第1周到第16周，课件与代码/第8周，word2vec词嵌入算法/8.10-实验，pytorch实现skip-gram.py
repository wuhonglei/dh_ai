import torch
import torch.nn as nn

# 定义SkipGram模型类，它继承nn.Module类
class SkipGram(nn.Module):
    # 函数传入参数vocab_size和embedding_dim
    # 分别代表词表中词语的数量和词向量的维度
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        # 定义两个embeddings嵌入层
        # in_embedding用于将目标词转换为词向量
        self.in_embeddings = nn.Embedding(vocab_size, embed_size)
        # out_embedding用于表示所有上下文词的词向量
        self.out_embeddings = nn.Embedding(vocab_size, embed_size)

    # 前向传播函数，输入目标词target
    def forward(self, target):
        in_vec = self.in_embeddings(target)  # 将target转为词向量
        out_vecs = self.out_embeddings.weight  # 获取词表中全部词语的词向量
        # 计算向量in_vec乘以矩阵out_vecs的转置，得到目标词和全部词语的点积
        scores = torch.matmul(in_vec, out_vecs.t())
        # softmax层可以直接放在损失函数CrossEntropyLoss中实现
        # 因此就不在前向传播forward中显示的实现了。
        return scores #返回score

# 函数make_train_data传入raw_text
# 函数计算raw_text中包含的词语集合vocab
# 词语数量vocab_size
# 词语到索引的字典word2ix
# 索引到词语的字典ix2word
def stat_raw_text(raw_text):
    # 将raw_text中保存的词语，放到集合set中去重
    vocab = set(raw_text)  # 得到了词语的集合vocab
    vocab_size = len(vocab)  # 计算词语集合的长度vocab_size
    word2ix = dict()  # 设置词语到索引的字典
    for ix, word in enumerate(vocab):  # 遍历词表vocab构造字典
        word2ix[word] = ix
    ix2word = dict()  # 设置索引到词语的字典
    for ix, word in enumerate(vocab):  # 遍历词表vocab构造字典
        ix2word[ix] = word
    return vocab, vocab_size, word2ix, ix2word


# 函数make_train_data传入raw_text
# 将raw_text构造为上下文的训练数据
def make_train_data(raw_text):
    data = []  # 保存训练数据
    window = 2  # 设置上下文窗口window = 2
    # 遍历raw_text
    for i in range(window, len(raw_text) - window):
        # 构造上下文，保存在context中
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]  # 目标词保存在target中
        # 将context和target一起添加到data
        data.append((context, target))
    return data  # 返回data


# 将词语word转为索引下标张量
def word_to_idx_tensor(word):
    # 使用word2ix[word]，获取词语对应的索引，然后将它转为张量返回
    return torch.tensor([word2ix[word]], dtype=torch.long)



if __name__ == '__main__':
    # 为了验证算法的正确性，准备一小段文本就可以了
    # 将文本通过split进行切词，切词结果保存在raw_text中
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # 使用两个函数stat_raw_text和make_train_data，处理raw_text
    vocab, vocab_size, word2ix, ix2word = stat_raw_text(raw_text)
    data = make_train_data(raw_text)

    # 在训练中，使用构造好的数据data，其中保存了上下文数据与目标词
    embedding_dim = 100 # 保存词向量的维度

    # 定义skip-gram模型，传入词语数量vocab_size和词向量维度embedding_dim
    model = SkipGram(vocab_size, embedding_dim)
    loss = nn.CrossEntropyLoss()  # 定义交叉熵误差
    # 优化器optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


    for epoch in range(200):  # 进入模型的循环迭代
        total_loss = 0  # 保存一轮训练的总损失值
        # 遍历data中保存的上下文context和目标词target
        for context, target in data:
            # 将目标词target转为索引形式
            target_idx = word_to_idx_tensor(target)
            # 使用模型model，预测目标词，得到结果out
            out = model(target_idx)
            for label in context: # 遍历全部的上下文词
                # 将每个上下文词都转为索引，保存在label_idx中
                label_idx = word_to_idx_tensor(label)
                # 将预测结果out和标签label_idx传入损失函数loss
                # 将损失值累加到total_loss中
                total_loss += loss(out, label_idx)

        total_loss.backward()  # 计算损失函数关于模型参数的梯度
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 将梯度清零


    # 遍历data中保存的上下文context和目标词target
    for context, target in data:
        # 将目标词target转为索引形式
        target_idx = word_to_idx_tensor(target)
        # 将target_idxs传入model，计算输出out
        out = model(target_idx)
        # 将输出，通过ix2word，转为预测词predict
        predict = ix2word[torch.argmax(out[0]).item()]
        # 将上下文词context、目标词target和预测词predict打印出来
        print(f'Context: {context}')
        print(f'Target: {target}')
        print(f'Prediction: {predict}\n')



