import torch
import torch.nn as nn
from torchinfo import summary
from gensim.models import KeyedVectors


class TextClassifier(nn.Module):
    """
    新闻文本分类器
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, padding_idx, word2vec_embeddings=None):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        if word2vec_embeddings is not None:
            self.embedding.weight.data.copy_(word2vec_embeddings)
            self.embedding.weight.requires_grad = False  # 冻结嵌入层参数
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim], seq_len 表示句子长度
        x = self.embedding(x)
        x = torch.sum(x, dim=1)  # x: [batch_size, embedding_dim]
        x = self.classifier(x)
        return x


def build_word2vec_embeddings(word_to_idx: dict[str, int], word_to_vector: KeyedVectors):
    """
    根据 word2vec 词向量构建词嵌入矩阵
    """
    vocab_size = len(word_to_idx)
    embed_dim = word_to_vector.vector_size
    embeddings = torch.zeros(vocab_size, embed_dim, dtype=torch.float)
    for word, idx in word_to_idx.items():
        if word in word_to_vector:
            embeddings[idx] = torch.tensor(word_to_vector[word])
    return embeddings


if __name__ == '__main__':
    model = TextClassifier(55592, 128, 20, 0)
    summary(model, input_data=torch.randint(0, 55592, (4, 85)),
            device='cpu', col_names=("input_size", "output_size", "num_params"))
