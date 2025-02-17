import numpy as np


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
