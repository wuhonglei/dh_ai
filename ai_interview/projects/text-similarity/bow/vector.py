from vocab import Vocab
from typing import List, Tuple
import pandas as pd
import time
import numpy as np
from numpy.typing import NDArray


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
        np_embeddings = np.array(embeddings, dtype=np.float16)
        total_words = np.sum(np_embeddings)
        if total_words == 0:
            return np_embeddings

        np_embeddings = np_embeddings / total_words
        l2 = np.linalg.norm(np_embeddings)
        return np_embeddings / l2

    def vectorize(self, title: str, content: str) -> Tuple[NDArray[np.float16], NDArray[np.float16]]:
        title_embeddings = self.vectorize_text(title)
        content_embeddings = self.vectorize_text(content)
        return title_embeddings, content_embeddings

    def vectorize_text(self, text: str) -> NDArray[np.float16]:
        words = self.vocab.tokenize(text, use_stop_words=True)
        indices = self.vocab.batch_encoder(words)
        embeddings = self.indices_to_embeddings(indices)
        embeddings = self.l2_normalize(embeddings)
        return embeddings

    def batch_vectorize_text(self, texts: List[str]) -> List[NDArray[np.float16]]:
        return [self.vectorize_text(text) for text in texts]


if __name__ == "__main__":
    vocab = Vocab()
    start_time = time.time()
    vocab.load_vocab_from_txt()
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
