"""
vocab类，用于构建和存储词汇表
"""

import os
import pandas as pd
from typing import Callable
from collections import Counter
from tqdm import tqdm

from config import columns, train_csv_path, vocab_dir


class Vocab:
    def __init__(self):
        self.counter = Counter()
        self.word_to_id = {
            '<pad>': 0,  # 填充词
            '<unk>': 1,  # 未知词
        }
        self.id_to_word = ['<pad>', '<unk>']
        self.padding_idx = 0
        self.unknown_idx = 1

    def build_vocab(self, csv_path: str, column_name: str, wordNgrams: int, tokenizer: Callable):
        data = pd.read_csv(csv_path)
        for text in data[column_name]:
            tokens = tokenizer(text)
            self.counter.update(tokens)
            if len(tokens) < wordNgrams:
                continue
            ngrams = []
            for i in range(len(tokens) - wordNgrams + 1):
                ngram = '_'.join(tokens[i:i + wordNgrams])
                ngrams.append(ngram)
            self.counter.update(ngrams)

    def save_vocab_freq(self, vocab_path: str):
        vocab_freq = {
            'word': [],
            'freq': []
        }
        for word, freq in self.counter.most_common():
            vocab_freq['word'].append(word)
            vocab_freq['freq'].append(freq)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        pd.DataFrame(vocab_freq).to_csv(vocab_path, index=False)

    def load_vocab_freq(self, vocab_path: str, min_freq: int, max_size: int | None = None):
        vocab_freq = pd.read_csv(vocab_path)
        for word, freq in zip(vocab_freq['word'], vocab_freq['freq']):
            if freq >= min_freq:
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word.append(word)
                if max_size and len(self.id_to_word) >= max_size:
                    break
            else:
                break
        return self.word_to_id, self.id_to_word

    def __len__(self):
        return len(self.word_to_id)

    def __getitem__(self, word: str):
        return self.word_to_id[word]

    def __contains__(self, word: str):
        return word in self.word_to_id


def build_vocab(wordNgrams: int):
    for column in tqdm(columns, desc='构建词汇表'):
        vocab_path = os.path.join(vocab_dir, f'{column}.csv')
        vocab = Vocab()
        vocab.build_vocab(train_csv_path, column,
                          wordNgrams, lambda x: x.split())
        vocab.save_vocab_freq(vocab_path)


def load_vocab():
    for column in tqdm(columns, desc='加载词汇表'):
        vocab_path = os.path.join(vocab_dir, f'{column}.csv')
        vocab = Vocab()
        vocab.load_vocab_freq(vocab_path, min_freq=5, max_size=None)
        return vocab


if __name__ == '__main__':
    build_vocab(wordNgrams=2)
