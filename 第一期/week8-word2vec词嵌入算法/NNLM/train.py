import os


import torch
import jieba
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from model import NNLM

from typing import Union

# 下载停用词列表（只需运行一次）
nltk.download('stopwords')

# 获取中文停用词列表
stopwords_list = stopwords.words('chinese')


def read_dir(directory: str):
    stop_words = set(['、', '：', '。', '，', '的', '等', '一', '二',
                     '三', '（', '）', '《', '》', '“', '”', ' ', '\n', '；', '*', '-', '.', '1', '2', '3', '4', '5']+stopwords_list)

    corpus: list[str] = []
    flatten_corpus: list[str] = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            for line in f:
                words = [word for word in jieba.cut(
                    line.strip()) if word not in stop_words]
                if words:
                    corpus.append(' '.join(words))
                    flatten_corpus.extend(words)
    return corpus, flatten_corpus


def stat_raw_text(raw_text: list[str]):
    vocab = set(raw_text)
    vocab_size = len(vocab)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}
    return vocab, vocab_size, word2idx, idx2word


def make_train_data(raw_text: list[str], context_size: int):
    data = []
    for line in raw_text:
        words = line.split()
        if len(words) < context_size + 1:
            continue

        for i in range(context_size, len(words)):
            context = words[i-context_size:i]
            target = words[i]
            data.append((context, target))

    return data


corpus, flatten_corpus = read_dir('data')
vacab, vocab_size, word2idx, idx2word = stat_raw_text(flatten_corpus)

context_size = 1
embedding_dim = 100     # 词嵌入维度
hidden_dim = 128        # 隐藏层维度
epochs = 100

train_data = make_train_data(corpus, context_size)

model = NNLM(vocab_size, embedding_dim, hidden_dim, context_size)
if os.path.exists('nnlm.pth'):
    checkpoint = torch.load('nnlm.pth')
    print('Load model from nnlm.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

critertion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    model.train()
    total_loss = torch.tensor(0.0)
    for context, target in train_data:
        context = torch.LongTensor([word2idx[word] for word in context])
        target = torch.LongTensor([word2idx[target]])

        optimizer.zero_grad()
        output = model(context)
        total_loss += critertion(output, target)

    total_loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {total_loss.item()}')

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'context_size': context_size,
    'hidden_dim': hidden_dim,
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'word2idx': word2idx,
    'idx2word': idx2word
}, 'nnlm.pth')
