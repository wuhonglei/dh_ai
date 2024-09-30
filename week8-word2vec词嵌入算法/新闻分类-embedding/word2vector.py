import os
from gensim.models import Word2Vec
from collections import defaultdict

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from dataset import NewsDataset, ReIterableSentences, build_vocab, collate_batch

# 1. 加载数据集
newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'), data_home='/mnt/model/nlp/scikit_learn_data/' if os.path.exists('/mnt') else None)

texts = newsgroups.data   # type: ignore
labels = newsgroups.target  # type: ignore

# 3. 数据分割
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 创建数据集
train_dataset = NewsDataset(train_texts, train_labels)
test_dataset = NewsDataset(test_texts, test_labels)
vocab = build_vocab(train_dataset)
# 遍历 vocab 中的词
sentences: list[list[str]] = []
for sentence, _ in train_dataset:
    new_sentence = []
    for word in sentence:
        if word in vocab:
            new_sentence.append(word)
    if new_sentence:
        sentences.append(new_sentence)

pass
# model = Word2Vec(sentences, sg=1,
#                  vector_size=100, window=5, min_count=1, workers=4)

# # 保存模型
# model.save('./models/word2vec.model')

# # 加载模型
# # model = Word2Vec.load('./models/word2vec.model')
# similar_words = model.wv.most_similar('dog')
# print(similar_words)
