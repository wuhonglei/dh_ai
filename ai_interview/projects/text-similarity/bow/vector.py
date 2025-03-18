from vocab import Vocab
import jieba
from typing import List
import pandas as pd
import time
from collections import Counter
import numpy as np


class Vector:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.empty_embedding = [0] * len(vocab)

    def indices_to_embeddings(self, indices: List[int]):
        """ 将索引转换为词向量 """
        embeddings = self.empty_embedding.copy()
        for index in indices:
            embeddings[index] += 1
        return embeddings

    def l2_normalize(self, embeddings: List[int]):
        np_embeddings = np.array(embeddings)
        l2 = np.linalg.norm(np_embeddings)
        if l2 == 0:
            return np_embeddings
        return np_embeddings / l2

    def vectorize(self, title: str, content: str):
        title_words = jieba.lcut(title)
        content_words = jieba.lcut(content)
        title_indices = self.vocab.batch_encoder(title_words)
        content_indices = self.vocab.batch_encoder(content_words)
        title_embeddings = self.indices_to_embeddings(title_indices)
        content_embeddings = self.indices_to_embeddings(content_indices)
        # 向量归一化
        title_embeddings = self.l2_normalize(title_embeddings)
        content_embeddings = self.l2_normalize(content_embeddings)
        return title_embeddings, content_embeddings

    def batch_vectorize(self, titles: List[str], contents: List[str]):
        for title, content in zip(titles, contents):
            title_embeddings, content_embeddings = self.vectorize(
                title, content)
            yield title_embeddings, content_embeddings

    def vectorize_title(self, title: str):
        title_words = jieba.lcut(title)
        title_indices = self.vocab.batch_encoder(title_words)
        return title_indices

    def vectorize_content(self, content: str):
        content_words = jieba.lcut(content)
        content_indices = self.vocab.batch_encoder(content_words)
        return content_indices

    def batch_vectorize_title(self, titles: List[str]):
        for title in titles:
            yield self.vectorize_title(title)

    def batch_vectorize_content(self, contents: List[str]):
        for content in contents:
            yield self.vectorize_content(content)


if __name__ == "__main__":
    vocab = Vocab()
    start_time = time.time()
    vocab.load_vocab_from_txt("../data/vocab.txt", min_freq=90)
    end_time = time.time()
    print(f"Vocab loading time: {end_time - start_time} seconds")
    vector = Vector(vocab)
    df = pd.read_csv("../data/val.csv")
    for i in range(1):
        row = df.iloc[i]
        title = row["title"]
        content = row["content"]
        title_embeddings, content_embeddings = vector.vectorize(title, content)
        print(len(title_embeddings))
        print(len(content_embeddings))
