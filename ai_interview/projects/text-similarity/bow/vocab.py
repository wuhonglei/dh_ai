import jieba
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, List
from collections import Counter, defaultdict
from dataset import NewsDatasetCsv
from utils.common import load_txt_file, load_json_file, timer_decorator, write_json_file, word_idf
import re


from config import DATASET_CONFIG, VOCAB_CONFIG, VocabConfig


class Vocab:
    def __init__(self, vocab_config: VocabConfig | None = None):
        self.vocab_config = vocab_config or VOCAB_CONFIG
        self.counter: Counter = Counter()
        self.word_to_index, self.index_to_word = self.initialize_word_and_index()
        self.word_to_idf: Dict[str, float] = {}
        self.idf_embedding: nn.Embedding | None = None
        # 将列表改为集合，提高查找效率
        self.low_freq_words: set[str] = set()
        self.high_freq_words: set[str] = set()
        # 创建一个过滤词集合，包含所有需要过滤的词
        self.ignored_words: set[str] = set()
        self.stop_words = self.load_stop_words()

    def initialize_word_and_index(self):
        special_tokens = ['<unk>', '<pad>']
        word_to_index: Dict[str, int] = {}
        index_to_word: List[str] = []
        for token in special_tokens:
            word_to_index[token] = len(word_to_index)
            index_to_word.append(token)

        return word_to_index, index_to_word

    def build_vocab_from_dataset(self, dataset: NewsDatasetCsv):
        self.counter = Counter()
        progress = tqdm(range(len(dataset)), desc="Building vocab")
        for i in progress:
            item = dataset[i]
            # 这里构建的是原始词汇表，所以不使用停用词
            self.counter.update(self.tokenize(
                item['title'], filter_word=False))
            self.counter.update(self.tokenize(
                item['content'], filter_word=True))
        return self.counter

    def tokenize(self, text: str, filter_word: bool = True) -> List[str]:
        token_list = jieba.lcut(text)
        if not filter_word:
            return token_list

        filtered_token_list = []
        for token in token_list:
            # 使用单次查找替代多次查找
            if token in self.ignored_words:
                continue
            filtered_token_list.append(token)
        return filtered_token_list

    def load_stop_words(self) -> set[str]:  # 返回类型改为 set
        if not self.vocab_config.use_stop_words:
            return set()
        stop_words: set[str] = set()
        for path in self.vocab_config.stop_words_paths:
            stop_words.update(load_txt_file(path))
        self.ignored_words.update(stop_words)  # 更新过滤词集合
        return stop_words

    def invalid_word(self, word: str) -> bool:
        return re.match(r'^[\w\s]+$', word) is None

    def load_vocab_from_txt(self):
        total_words = 0
        with open(self.vocab_config.vocab_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue

                word, freq = line.rsplit(' ', 1)  # 避免空格词汇被错误分割
                total_words += 1
                if word in self.stop_words:
                    continue

                if self.vocab_config.max_freq and int(freq) >= self.vocab_config.max_freq:
                    self.high_freq_words.add(word)
                    self.ignored_words.add(word)
                    continue

                if int(freq) >= self.vocab_config.min_freq:
                    self.word_to_index[word] = len(self.word_to_index)
                    self.index_to_word.append(word)
                else:
                    self.low_freq_words.add(word)
                    self.ignored_words.add(word)
        print(
            f"Total words: {total_words}, Used words: {len(self.index_to_word)}, low freq words: {len(self.low_freq_words)}, high freq words: {len(self.high_freq_words)}")
        self.vocab = set(self.index_to_word)

    def __len__(self):
        return len(self.word_to_index)

    def __getitem__(self, word: str):
        return self.word_to_index[word]

    def encoder(self, word: str):
        if word not in self.word_to_index:
            return self.word_to_index['<unk>']
        return self.word_to_index[word]

    def batch_encoder(self, words: List[str]):
        return [self.encoder(word) for word in words]

    def decoder(self, index: int):
        return self.index_to_word[index]

    def batch_decoder(self, indices: List[int]):
        return [self.decoder(index) for index in indices]

    @timer_decorator
    def build_word_doc_counts(self, dataset: NewsDatasetCsv):
        """ 构建单词在文档中的出现次数 """
        word_doc_counts: Dict[str, int] = defaultdict(int)
        total_doc_count = len(dataset)
        progress = tqdm(range(total_doc_count),
                        desc="Building word doc counts")
        for i in progress:
            content = dataset[i]['content']
            words = set(self.tokenize(content, filter_word=False))
            for word in words:
                word_doc_counts[word] += 1
        return word_doc_counts

    def save_vocab_set(self, path: str, min_freq: int):
        with open(path, 'w') as f:
            used = 0
            for word, freq in self.counter.most_common():
                if freq >= min_freq:
                    f.write(f"{word} {freq}\n")
                    used += 1
                else:
                    break

        print(f"Total words: {self.counter.total()}, Used words: {used}")


def save_vocab_txt():
    """
    构建词汇表，不过滤词频，并保存到 txt 文件中
    """
    vocab = Vocab()
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    vocab.build_vocab_from_dataset(dataset)
    vocab.save_vocab_set(VOCAB_CONFIG.vocab_path, 0)


def save_vocab_idf():
    """
    构建词汇表，并计算 idf 值，并保存到 txt 文件中
    """

    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    word_doc_counts = load_json_file(
        VOCAB_CONFIG.word_counts_path)
    total_doc_count = len(dataset)
    result: Dict[str, float] = {}
    for word, count in word_doc_counts.items():
        result[word] = word_idf(count, total_doc_count)
    write_json_file(VOCAB_CONFIG.word_idf_path, result)


if __name__ == "__main__":
    save_vocab_idf()
