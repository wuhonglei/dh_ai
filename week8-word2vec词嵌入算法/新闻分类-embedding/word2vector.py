import os
import time
from gensim.models import Word2Vec

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from dataset import NewsDataset, build_vocab

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
train_dataset = NewsDataset(train_texts, train_labels, use_cache=True)
test_dataset = NewsDataset(test_texts, test_labels, use_cache=True)
special_tokens = ['<pad>', '<unk>']
vocab = build_vocab(train_dataset, special_tokens)
print('vocab size:', len(vocab))

# 遍历 vocab 中的词
sentences: list[list[str]] = train_dataset.get_sentence_in_vocab(vocab)

# 训练 Word2Vec 模型
start_time = time.time()
model = Word2Vec(sentences, sg=1,
                 vector_size=100, window=5, min_count=1, workers=4)
print(f"Training time: {time.time() - start_time}")

# 保存模型
model.save('./models/word2vec.model')

# # 加载模型
# # model = Word2Vec.load('./models/word2vec.model')
# similar_words = model.wv.most_similar('dog')
# print(similar_words)
