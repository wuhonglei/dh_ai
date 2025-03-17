import jieba
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from encoder import Encoder


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text_list = [line.strip()
                         for line in f.readlines() if line.strip()]
        return text_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [word.strip() for word in jieba.cut(self.data[idx]) if word.strip()]


def build_vocab(dataset: TextDataset):
    vocab = set()
    for sentence in dataset:
        for word in sentence:
            vocab.add(word)
    return vocab


def write_vocab(vocab, path):
    special_tokens = [
        '<pad>',
        '<s>',
        '</s>',
        '<unk>',
    ]
    vocab = special_tokens + list(vocab)

    with open(path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')


def collate_fn(batch, encoder: Encoder, pad_token_id=0):
    input_ids = pad_sequence([torch.tensor(encoder.encode(sentence))
                              for sentence in batch], batch_first=True, padding_value=pad_token_id)
    labels = input_ids.clone()
    input_ids = torch.roll(input_ids, shifts=-1, dims=-1)
    input_ids[:, -1] = pad_token_id
    return input_ids, labels


if __name__ == "__main__":
    dataset = TextDataset("data/train.txt")
    vocab = build_vocab(dataset)
    print(len(vocab))
    write_vocab(vocab, "data/vocab.txt")
