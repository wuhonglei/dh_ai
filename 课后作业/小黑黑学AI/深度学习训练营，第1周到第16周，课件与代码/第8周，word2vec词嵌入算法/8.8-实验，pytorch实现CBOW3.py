import torch
import torch.nn as nn

# 定义CBOW模型类，它继承nn.Module类
class CBOW(nn.Module):
    # 类的初始化函数init
    # 函数传入参数vocab_size和embedding_dim
    # 分别代表词表中词语的数量和词向量的维度
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        # 设置词嵌入层embeddings，它可以直接使用nn.Embedding构造出来
        # 它是vocab_size乘embedding_dim的词嵌入矩阵
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 设置线性层linear，它的隐藏层中包括embedding_dim个神经元
        # 输出层包括vocab_size个神经元
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # 函数传入上下文词语的索引context_idxs
    def forward(self, context_idxs):
        # 首先使用embeddings层计算每个词语的词向量
        embeds = self.embeddings(context_idxs)
        # 将这些词向量求和，生成一个统一的上下文表示embeds
        embeds = sum(embeds).view(1, -1)
        # 使用线性层，计算输出out
        out = self.linear(embeds)
        # softmax层可以直接放在损失函数CrossEntropyLoss中实现
        # 因此就不在前向传播forward中显示的实现了
        return out  # 返回out


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


# 实现将上下文context转为索引下标的函数context_to_idxs
def context_to_idxs(context, word2ix):
    idxs = list()
    for w in context:  # 遍历上下文context
        # 将每个词的索引值word2ix[w]，保存到idxs列表
        idxs.append(word2ix[w])
    # 返回idxs列表的张量形式
    return torch.tensor(idxs, dtype=torch.long)


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
    # 定义CBOW模型，传入词语数量vocab_size和词向量维度embedding_dim
    model = CBOW(vocab_size, embedding_dim)
    loss = nn.CrossEntropyLoss() #定义交叉熵误差
    # 优化器optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


    for epoch in range(50): #进入模型的循环迭代
        total_loss = 0 #保存一轮训练的总损失值
        # 遍历data中保存的上下文context和目标词target
        for context, target in data:
            # 使用函数context_to_vector，将上下文转为索引形式
            context_idxs = context_to_idxs(context, word2ix)
            # 使用模型model，预测上下文，得到结果out
            out = model(context_idxs)
            # 使用字典word2ix，找到目标词target的标签label
            label = torch.tensor([word2ix[target]])
            # 将预测结果out和标签label传入损失函数loss
            # 将损失值累加到total_loss中
            total_loss += loss(out, label)

        total_loss.backward() #计算损失函数关于模型参数的梯度
        optimizer.step() #更新模型参数
        optimizer.zero_grad() #将梯度清零


    # 遍历data中保存的上下文context和目标词target
    for context, target in data:
        # 将context转为上下文词语的索引形式
        context_idxs = context_to_idxs(context, word2ix)
        # 将context_idxs传入model，计算输出out
        out = model(context_idxs)
        # 将输出，通过ix2word，转为预测词predict
        predict = ix2word[torch.argmax(out[0]).item()]
        # 将上下文词context、目标词target和预测词predict打印出来
        print(f'Context: {context}')
        print(f'Target: {target}')
        print(f'Prediction: {predict}\n')


