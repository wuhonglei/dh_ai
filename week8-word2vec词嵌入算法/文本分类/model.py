import torch
import torch.nn as nn

# 定义神经网络


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, padding_idx):
        super(TextClassifier, self).__init__()
        # 定义一个词嵌入层
        # embedding层的大小是vocab_size×embed_dim，表示词表大小乘词向量维度
        # 例如，5000×128
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        # 定义一个线性层
        # 线性层的大小是embed_dim×num_classes，表示词向量维度乘类别数量
        # 例如，128×3
        self.fc = nn.Linear(embed_dim, num_classes)

    # 在前向传播，forward函数中，函数的输入是文本序列x
    # 例如，输入一个2×6的文本序列x，表示2个样本，每个样本6个单词
    def forward(self, x):
        # 将每个单词都转为词向量
        # 例如，输入的x，是2×6的，每个单词用128维表示
        # 那么x经过embedding层后，就会变为2×6×128的结果
        x = self.embedding(x)

        # 计算一个句子中各个单词的词向量和
        # 也就是将每个句子中单词的词向量相加后，得到每个句子的向量表示
        # 例如，2×6×128的x，就会被转换为2×128的结果
        # 也就是2个句子，每个句子都用128维来表示
        x = torch.sum(x, dim=1)

        x = self.fc(x)  # 最终的分类结果
        # 例如，2×128的2个句子，就会计算为2×3的结果
        # 这里的3就代表了3种类别

        return x  # 将其返回

# 观察输入x，经过神经网络计算后的变化


def print_forward(model, x):
    print(f"x: {x.shape}")
    x = model.embedding(x)
    print(f"after embedding: {x.shape}")
    x = torch.sum(x, dim=1)
    print(f"after sum: {x.shape}")
    output = model.fc(x)
    print(f"after fc: {output.shape}")
    print(output)


if __name__ == "__main__":
    vocab_size = 5000  # 词汇表大小
    embed_dim = 4  # 词嵌入维度
    num_classes = 3  # 分类类别数量
    padding_idx = 1  # 假设填充索引为0

    # 创建带有embedding层的神经网络
    model = TextClassifier(vocab_size, embed_dim, num_classes, padding_idx)

    # 创建一个2×6大小的输入数据x，其中包含填充值
    x = torch.tensor([[3, 5, 1, 1, 1, 1],
                      [3, 4, 5, 6, 1, 1]])

    print_forward(model, x)  # 打印x在神经网络中的变化

    output = model(x)  # 计算结果output
    print("Output Shape:", output.shape)
