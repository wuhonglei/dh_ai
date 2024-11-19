import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import pickle
from torch.nn.utils.rnn import pad_sequence
from utils.model import create_pad_mask, create_subsequent_mask
from utils.common import get_file_name, exists_cache, save_cache, load_cache


def split_token(text: str):
    return [token for token in text.lower().split() if token]


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
    def __init__(self, csv_path: str, use_cache=True):
        cache_name = f'./cache/csv/{get_file_name(csv_path)}.pkl'
        if use_cache and exists_cache(cache_name):
            self.src, self.target = load_cache(cache_name)
            return

        df = pd.read_csv(csv_path, sep="\t")
        self.src = df['en'].apply(lambda x: split_token(x))
        self.target = df['zh'].apply(
            lambda x: ['<sos>'] + split_token(x) + ['<eos>'])

        save_cache(cache_name, (self.src, self.target))

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


def build_vocab(dataset: TranslateDataset, use_cache=True):
    cache_name = f'./cache/vocab/{len(dataset)}.pkl'
    if use_cache and exists_cache(cache_name):
        return load_cache(cache_name)

    src_word_list = []
    target_word_list = []
    for src, target in zip(dataset.src, dataset.target):
        src_word_list.extend(src)
        target_word_list.extend(target)

    src_vocab = build_vocab_from_list(src_word_list)
    target_vocab = build_vocab_from_list(target_word_list)
    save_cache(cache_name, (src_vocab, target_vocab))

    return src_vocab, target_vocab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, src_vocab, target_vocab):
    src = list()
    target = list()
    for src_sample, target_sample in batch:
        src_tokens = [src_vocab[token] for token in src_sample]
        target_tokens = [target_vocab[token] for token in target_sample]

        src.append(torch.LongTensor(src_tokens))
        target.append(torch.LongTensor(target_tokens))

    src_pad_index = src_vocab['<pad>']
    tgt_pad_index = target_vocab['<pad>']
    src_batch = pad_sequence(
        src, padding_value=src_pad_index, batch_first=True)
    target_batch = pad_sequence(
        target, padding_value=tgt_pad_index, batch_first=True)

    # 创建掩码
    # [batch_size, 1, 1, src_seq_len]
    src_mask = create_pad_mask(src_batch, src_pad_index).to(device)
    tgt_pad_mask = create_pad_mask(target_batch, tgt_pad_index).to(device)
    tgt_sub_mask = create_subsequent_mask(target_batch.size(1)).to(
        device)  # [tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)  # 合并掩码

    return src_batch, target_batch, src_mask, tgt_mask


if __name__ == '__main__':
    for name in ['train', 'test', 'validation']:
        dataset = TranslateDataset(f'csv/{name}.csv')
        src_vocab, target_vocab = build_vocab(dataset)
        with open(f'cache/{name}/src_vocab.pkl', 'wb') as f:
            pickle.dump(src_vocab, f)
        with open(f'cache/{name}/target_vocab.pkl', 'wb') as f:
            pickle.dump(target_vocab, f)
