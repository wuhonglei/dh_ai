import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import pickle
from torch.nn.utils.rnn import pad_sequence
import time


def split_token(text: str):
    return [token.lower() for token in text.split() if token]


class Vocab():
    def __init__(self, vocab_dict: dict[str, int]):
        self.word_to_index = vocab_dict
        index_to_word = {}
        for word, index in vocab_dict.items():
            index_to_word[index] = word
        self.index_to_word = index_to_word

    def __len__(self):
        return len(self.word_to_index)

    def get_itos(self):
        return self.index_to_word

    def get_stoi(self):
        return self.word_to_index

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word_to_index.get(key, self.word_to_index['<unk>'])
        elif isinstance(key, int):
            return self.index_to_word.get(key, '<unk>')


class TranslateDataset(Dataset):
    def __init__(self, csv_path: str, ):
        df = pd.read_csv(csv_path, sep="\t")
        self.src = df['en'].apply(
            lambda x: ['<sos>'] + split_token(x) + ['<eos>'])
        self.target = df['zh'].apply(
            lambda x: ['<sos>'] + split_token(x) + ['<eos>'])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.target[idx]


def build_vocab_from_list(word_list: list[str]):
    counter = Counter(word_list)
    vocab = {
        '<unk>': 0,
        '<pad>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    for i, word in enumerate(counter):
        if word not in vocab:
            vocab[word] = len(vocab)
    return Vocab(vocab)


def build_vocab(dataset: TranslateDataset):
    src_word_list = []
    target_word_list = []
    for src, target in zip(dataset.src, dataset.target):
        src_word_list.extend(src)
        target_word_list.extend(target)

    src_vocab = build_vocab_from_list(src_word_list)
    target_vocab = build_vocab_from_list(target_word_list)

    return src_vocab, target_vocab


def collate_fn(batch, src_vocab, target_vocab):
    start_time = time.time()
    src = list()
    target = list()
    for src_sample, target_sample in batch:
        src_tokens = [src_vocab[token]for token in src_sample]
        target_tokens = [target_vocab[token] for token in target_sample]

        src.append(torch.LongTensor(src_tokens))
        target.append(torch.LongTensor(target_tokens))

    src_batch = pad_sequence(
        src, padding_value=src_vocab['<pad>'], batch_first=True)
    target_batch = pad_sequence(
        target, padding_value=target_vocab['<pad>'], batch_first=True)

    # print(f'collate_fn time: {time.time() - start_time}')
    return src_batch, target_batch


if __name__ == '__main__':
    for name in ['train', 'test', 'validation']:
        dataset = TranslateDataset(f'csv/{name}.csv')
        src_vocab, target_vocab = build_vocab(dataset)
        with open(f'cache/{name}/src_vocab.pkl', 'wb') as f:
            pickle.dump(src_vocab, f)
        with open(f'cache/{name}/target_vocab.pkl', 'wb') as f:
            pickle.dump(target_vocab, f)
