from vocab import Vocab
import jieba
from typing import List


class Vector:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def vectorize(self, title: str, content: str):
        title_words = jieba.lcut(title)
        content_words = jieba.lcut(content)
        title_indices = self.vocab.batch_encoder(title_words)
        content_indices = self.vocab.batch_encoder(content_words)
        return title_indices, content_indices

    def batch_vectorize(self, titles: List[str], contents: List[str]):
        for title, content in zip(titles, contents):
            title_indices, content_indices = self.vectorize(title, content)
            yield title_indices, content_indices

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
