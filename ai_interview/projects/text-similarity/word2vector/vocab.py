import jieba
from tqdm import tqdm
from typing import Dict, List
from collections import Counter
from dataset import NewsDatasetCsv
from utils.common import load_txt_file


from config import DATASET_CONFIG, VOCAB_CONFIG, VocabConfig


class Vocab:
    def __init__(self, vocab_config: VocabConfig | None = None):
        self.vocab_config = vocab_config or VOCAB_CONFIG
        self.counter: Counter = Counter()
        self.word_to_index, self.index_to_word = self.initialize_word_and_index()
        self.pad_idx = self.word_to_index['<pad>']
        self.stop_words = self.load_stop_words()
        self.word_to_idf: Dict[str, float] = {}

        # 低频词和超高频词
        self.low_freq_words = []
        self.high_freq_words = []

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
                item['title'], use_stop_words=False))
            self.counter.update(self.tokenize(
                item['content'], use_stop_words=True))
        return self.counter

    def tokenize(self, text: str, use_stop_words: bool = True) -> List[str]:
        token_list = jieba.lcut(text)
        if not use_stop_words or not self.stop_words:
            return token_list

        token_list = []
        for token in token_list:
            """
            过滤 停用词、低频词和超高频词
            """
            if token in self.stop_words or token in self.high_freq_words or token in self.low_freq_words:
                continue
            token_list.append(token)
        return token_list

    def load_stop_words(self):
        if not self.vocab_config.use_stop_words:
            return []
        stop_words = set()
        for path in self.vocab_config.stop_words_paths:
            stop_words.update(load_txt_file(path))
        return list(stop_words)

    def load_vocab_from_txt(self):
        with open(self.vocab_config.vocab_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue

                word, freq = line.rsplit(' ', 1)  # 避免空格词汇被错误分割
                if word in self.stop_words:
                    continue

                if self.vocab_config.max_freq and int(freq) >= self.vocab_config.max_freq:
                    self.high_freq_words.append(word)
                    continue

                if int(freq) >= self.vocab_config.min_freq:
                    self.word_to_index[word] = len(self.word_to_index)
                    self.index_to_word.append(word)
                else:
                    self.low_freq_words.append(word)

        self.vocab = set(self.index_to_word)

    def __len__(self):
        return len(self.word_to_index)

    def __getitem__(self, word: str):
        return self.word_to_index[word]

    def encoder(self, word: str):
        if word not in self.word_to_index:
            return self.word_to_index['<unk>']
        return self.word_to_index[word]

    def batch_encoder(self, texts: List[str]):
        return [self.encoder(text) for text in texts]

    def decoder(self, index: int):
        return self.index_to_word[index]

    def batch_decoder(self, indices: List[int]):
        return [self.decoder(index) for index in indices]

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


if __name__ == "__main__":
    save_vocab_txt()
