from vocab import Vocab
from typing import List, Tuple
import pandas as pd
import time
import numpy as np
from numpy.typing import NDArray


class Vector:
    def __init__(self, vocab: Vocab, idf_dict: dict[str, float]):
        self.vocab = vocab
        self.empty_tf_embedding = [0.0] * len(vocab)
        self.empty_idf_embedding = [1.0] * len(vocab)
        self.idf_dict = idf_dict  # {'单词': idf} 值

    def indices_to_idf_embeddings(self, indices: List[int]):
        """ 将索引转换为词向量 (idf 值)"""
        embeddings = self.empty_idf_embedding.copy()
        unique_indices = set(indices)
        for index in unique_indices:
            word = self.vocab.decoder(index)
            if word in self.idf_dict:
                embeddings[index] = self.idf_dict[word]
        return embeddings

    def indices_to_tf_embeddings(self, indices: List[int]):
        """ 将索引转换为词向量 (原始 tf 值)"""
        embeddings = self.empty_tf_embedding.copy()
        for index in indices:
            embeddings[index] += 1
        return embeddings

    def l2_normalize(self, embeddings: NDArray[np.float16]):
        l2 = np.linalg.norm(embeddings)
        if l2 == 0:
            return embeddings
        return embeddings / l2

    def vectorize(self, title: str, content: str) -> Tuple[NDArray[np.float16], NDArray[np.float16]]:
        title_embeddings = self.vectorize_text(title)
        content_embeddings = self.vectorize_text(content)
        return title_embeddings, content_embeddings

    def vectorize_text(self, text: str, use_idf: bool = True) -> NDArray[np.float16]:
        words = self.vocab.tokenize(text, use_stop_words=True)
        indices = self.vocab.batch_encoder(words)
        tf_embeddings = self.indices_to_tf_embeddings(indices)
        if use_idf:
            idf_embeddings = self.indices_to_idf_embeddings(indices)
            tf_idf_embeddings = np.multiply(
                tf_embeddings, idf_embeddings, dtype=np.float16)
        else:
            tf_idf_embeddings = np.array(tf_embeddings, dtype=np.float16)
        tf_idf_embeddings = self.l2_normalize(tf_idf_embeddings)
        return tf_idf_embeddings

    def batch_vectorize_text(self, texts: List[str]) -> List[NDArray[np.float16]]:
        return [self.vectorize_text(text) for text in texts]


if __name__ == "__main__":
    vocab = Vocab()
    start_time = time.time()
    vocab.load_vocab_from_txt()
    end_time = time.time()
    print(f"Vocab loading time: {end_time - start_time} seconds")
    vector = Vector(vocab, {})
    df = pd.read_csv("../data/val.csv")
    for i in range(1):
        row = df.iloc[i]
        title = row["title"]
        content = row["content"]
        title_embeddings, content_embeddings = vector.vectorize(title, content)
        print(len(title_embeddings))
        print(len(content_embeddings))
