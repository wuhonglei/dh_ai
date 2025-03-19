import jieba
from tqdm import tqdm
from typing import Dict, List
from collections import Counter
from dataset import NewsDatasetCsv


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
            title, content = dataset[i]
            self.counter.update(jieba.lcut(title))
            self.counter.update(jieba.lcut(content))
        return self.counter

    def load_vocab_from_txt(self, path: str, min_freq: int = 90):
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

    def save_vocab_set(self, path: str, min_freq: int = 1):
        total_words = 0
        ignored_words = 0
        with open(path, 'w') as f:
            for word, freq in self.counter.most_common():
                total_words += 1
                if freq >= min_freq:
                    f.write(f"{word} {freq}\n")
                else:
                    ignored_words += 1

        print(f"Total words: {total_words}, Ignored words: {ignored_words}")


if __name__ == "__main__":
    vocab = Vocab()
    vocab.load_vocab_from_txt("../data/vocab.txt", min_freq=90)
    print(vocab.word_to_index)
    print(vocab.index_to_word)
