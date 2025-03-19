import jieba
from tqdm import tqdm
from typing import Dict, List
from collections import Counter
from dataset import NewsDatasetCsv
from config import DATASET_CONFIG


class Vocab:
    def __init__(self):
        self.counter: Counter = Counter()
        self.special_tokens = ['<unk>', '<pad>']
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: List[str] = []
        for token in self.special_tokens:
            self.word_to_index[token] = len(self.word_to_index)
            self.index_to_word.append(token)

    def build_vocab_from_dataset(self, dataset: NewsDatasetCsv):
        self.counter = Counter()
        progress = tqdm(range(len(dataset)), desc="Building vocab")
        for i in progress:
            item = dataset[i]
            self.counter.update(jieba.lcut(item['title']))
            self.counter.update(jieba.lcut(item['content']))
        return self.counter

    def load_vocab_from_txt(self, path: str, min_freq):
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue

                word, freq = line.rsplit(' ', 1)  # 避免空格词汇被错误分割
                if int(freq) >= min_freq:
                    self.word_to_index[word] = len(self.word_to_index)
                    self.index_to_word.append(word)
                else:
                    break

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


if __name__ == "__main__":
    vocab = Vocab()
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    vocab.build_vocab_from_dataset(dataset)
