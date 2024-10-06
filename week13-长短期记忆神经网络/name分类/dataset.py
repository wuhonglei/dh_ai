import torch
import os
import string
import re
import unicodedata
from torch.utils.data import Dataset


class NamesDataset(Dataset):
    def __init__(self, data_dir: str):
        self.names = []
        self.labels = []
        self.name2idx: dict[str, int] = {}
        self.idx2name: dict[int, str] = {}
        self.all_letters = string.ascii_letters + " .,;'"
        self.char2index = self.get_char_to_index(self.all_letters)
        self.pattern = re.compile(f'[^{self.all_letters}]')

        i = 0
        for filename in os.listdir(data_dir):
            if not filename.endswith('.txt'):
                continue
            name = filename.split('.')[0]
            self.name2idx[name] = i
            self.idx2name[i] = name
            label = self.name2idx[name]
            i += 1
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    self.names.append(
                        self.normalize_name(line.strip()))
                    self.labels.append(label)

    def normalize_name(self, name: str):
        nor_name = unicodedata.normalize('NFKD', name).encode(
            'ascii', 'ignore').decode('ascii')
        return self.pattern.sub('', nor_name)

    def get_char_to_index(self, all_letters: str):
        char2index = {char: i for i, char in enumerate(all_letters)}
        return char2index

    def name_to_tensor(self, name: str):
        # [len(name), 1, len(all_letters)]
        tensor = torch.zeros(len(name), 1, len(self.all_letters))
        for i, char in enumerate(name):
            tensor[i][0][self.char2index[char]] = 1  # one-hot
        return tensor

    def __len__(self):
        return len(self.names)

    def get_labels_num(self):
        return len(self.name2idx)

    def get_label_index(self, name: str):
        return self.name2idx.get(name, -1)

    def get_label_name(self, idx: int):
        return self.idx2name.get(idx, None)

    def get_label_names(self):
        return list(self.name2idx.keys())

    def __getitem__(self, idx):
        return self.names[idx], self.labels[idx]


if __name__ == '__main__':
    dataset = NamesDataset('data/names')
    print(dataset[0])
