import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np

# 定义 Vocabulary 类


class Vocabulary:
    def __init__(self, corpus, min_count=5):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}
        self.total_words = 0
        self.build_vocab(corpus, min_count)
        self.vocab_size = len(self.word2idx)
        self.word_probs = self.get_unigram_table()

    def build_vocab(self, corpus, min_count):
        word_counts = {}
        for line in corpus:
            for word in line.strip().split():
                word_counts[word] = word_counts.get(word, 0) + 1
                self.total_words += 1
        idx = 0
        for word, count in word_counts.items():
            if count >= min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.word_freq[idx] = count
                idx += 1

    def get_unigram_table(self):
        # 构建用于负采样的表
        power = 0.75
        norm = sum([freq ** power for freq in self.word_freq.values()])
        table_size = 1e8  # 根据需要调整
        table = []

        for idx in self.word_freq:
            prob = (self.word_freq[idx] ** power) / norm
            count = int(prob * table_size)
            table.extend([idx] * count)
        return np.array(table)

# 自定义 Dataset


class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=5, negative_samples=5):
        self.corpus = corpus
        self.vocab = vocab
        self.window_size = window_size
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        words = sentence.strip().split()
        word_indices = [self.vocab.word2idx[word]
                        for word in words if word in self.vocab.word2idx]
        pairs = []
        for i, center in enumerate(word_indices):
            window = random.randint(1, self.window_size)
            context_indices = word_indices[max(
                0, i - window): i] + word_indices[i + 1: i + window + 1]
            for context in context_indices:
                pairs.append((center, context))
        return pairs

    def collate_fn(self, batch):
        centers = []
        contexts = []
        negatives = []
        for pairs in batch:
            for center, context in pairs:
                centers.append(center)
                contexts.append(context)
                neg_samples = np.random.choice(
                    self.vocab.word_probs, size=self.negative_samples).tolist()
                negatives.append(neg_samples)
        return torch.LongTensor(centers), torch.LongTensor(contexts), torch.LongTensor(negatives)

# 定义 Skip-Gram 模型


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words, negative_words):
        center_embeds = self.in_embeddings(
            center_words)  # (batch_size, embedding_dim)
        context_embeds = self.out_embeddings(
            context_words)  # (batch_size, embedding_dim)
        # (batch_size, negative_samples, embedding_dim)
        neg_embeds = self.out_embeddings(negative_words)

        # 正样本得分
        pos_score = torch.mul(center_embeds, context_embeds).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        # 负样本得分
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze()
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

        # 总损失
        loss = - (pos_loss + neg_loss).mean()
        return loss


# 参数设置
embedding_dim = 100
batch_size = 2
epochs = 5
learning_rate = 0.01

# 读取大语料


def corpus_reader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# 假设语料文件路径为 'large_corpus.txt'
# corpus = corpus_reader('large_corpus.txt')


# 为演示，使用小语料代替
corpus = [
    "我们 都 是 好 朋友",
    "你们 也 是 我们 的 朋友",
    "他们 是 新 同学",
    "我们 欢迎 新 同学"
    # ... 更多句子
]

# 初始化词汇表和数据集
vocab = Vocabulary(corpus, min_count=1)
dataset = SkipGramDataset(corpus, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=dataset.collate_fn)

# 初始化模型和优化器
model = SkipGramModel(vocab.vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(1, epochs + 1):
    total_loss = 0
    for centers, contexts, negatives in dataloader:
        optimizer.zero_grad()
        loss = model(centers, contexts, negatives)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

# 保存模型
# torch.save(model.state_dict(), 'skipgram_model.pth')

# 词向量提取
# word_embeddings = model.in_embeddings.weight.data
