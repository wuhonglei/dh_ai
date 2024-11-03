import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import pickle
from torch.nn.utils.rnn import pad_sequence


class TranslateDataset(Dataset):
    def __init__(self, csv_path: str, ):
        df = pd.read_csv(csv_path, sep="\t")
        self.src = df['en'].apply(
            lambda x: ['<sos>'] + self.split_token(x) + ['<eos>'])
        self.target = df['zh'].apply(
            lambda x: ['<sos>'] + self.split_token(x) + ['<eos>'])

    def __len__(self):
        return len(self.src)

    def split_token(self, text: str):
        return [token for token in text.split() if token]

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
    return vocab


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
    src = list()
    target = list()
    for src_sample, target_sample in batch:
        src_tokens = [src_vocab.get(token, src_vocab['<unk>'])
                      for token in src_sample]
        target_tokens = [target_vocab.get(
            token, target_vocab['<unk>']) for token in target_sample]

        src.append(torch.LongTensor(src_tokens))
        target.append(torch.LongTensor(target_tokens))

    src_batch = pad_sequence(
        src, padding_value=src_vocab['<pad>'], batch_first=True)
    target_batch = pad_sequence(
        target, padding_value=target_vocab['<pad>'], batch_first=True)

    return src_batch, target_batch


if __name__ == '__main__':
    for name in ['train', 'test', 'validation']:
        dataset = TranslateDataset(f'csv/{name}.csv')
        src_vocab, target_vocab = build_vocab(dataset)
        with open(f'cache/{name}/src_vocab.pkl', 'wb') as f:
            pickle.dump(src_vocab, f)
        with open(f'cache/{name}/target_vocab.pkl', 'wb') as f:
            pickle.dump(target_vocab, f)
