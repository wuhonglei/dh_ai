import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import jieba
import nltk
from nltk.corpus import stopwords
import pickle
import json


from vocabulary import Vocabulary
from dataset import SkipGramDataset
from model import SkipGramModel


# 参数设置
embedding_dim = 200
batch_size = 512
epochs = 200
learning_rate = 0.01

# 下载停用词列表（只需运行一次）
nltk.download('stopwords')

# 获取中文停用词列表
stopwords_list = stopwords.words('chinese')


def read_dir(directory: str):
    stop_words = set(['、', '：', '。', '，', '的', '等', '一', '二',
                     '三', '（', '）', '《', '》', '“', '”', ' ', '\n', '；']+stopwords_list)

    corpus = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            for line in f:
                words = [word for word in jieba.cut(
                    line.strip()) if word not in stop_words]
                corpus.append(' '.join(words))
    return corpus


# 为演示，使用小语料代替
# corpus = [
#     "我们 都 是 好 朋友",
#     "你们 也 是 我们 的 朋友",
#     "他们 是 新 同学",
#     "我们 欢迎 新 同学"
#     # ... 更多句子
# ]

corpus = read_dir('./data')

# 初始化词汇表和数据集
vocab = Vocabulary(corpus, min_count=2)
dataset = SkipGramDataset(corpus, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=dataset.collate_fn, shuffle=True)

# 初始化模型和优化器
model = SkipGramModel(vocab.vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print('vocab.vocab_size', vocab.vocab_size)

# 训练
for epoch in range(1, epochs + 1):
    for i, batch in enumerate(dataloader):
        centers, contexts, negatives = batch
        optimizer.zero_grad()
        loss = model(centers, contexts, negatives)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab.vocab_size,
    'embedding_dim': embedding_dim,
    'word2idx': vocab.word2idx,
    'idx2word': vocab.idx2word
}, "./models/skip_gram.pth")

# 保存词汇表 pickle
with open('./models/vocab.pkl', 'wb') as f:
    data = {
        'word2idx': vocab.word2idx,
        'idx2word': vocab.idx2word
    }
    pickle.dump(data, f)

with open('./models/vocab.json', 'w') as f:
    data = {
        'word2idx': vocab.word2idx,
        'idx2word': vocab.idx2word
    }
    json.dump(data, f)
