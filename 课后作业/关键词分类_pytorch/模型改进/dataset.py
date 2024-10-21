import os

from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pickle

from tokennizer.sg import tokenize_sg
from tokennizer.my import tokenize_my
from tokennizer.th import tokenize_th
from tokennizer.tw import tokenize_tw

from typing import Sequence

token_dict = {
    'SG': tokenize_sg,
    'MY': tokenize_my,
    'TH': tokenize_th,
    'TW': tokenize_tw,
}


class KeywordCategoriesDataset(Dataset):
    def __init__(self, keywords: list[str], labels: list[str], country: str) -> None:
        unique_labels = list(set(labels))
        self.label2index = self.get_label_to_index(unique_labels)
        self.index2label = self.get_index_to_label(unique_labels)
        self.data = self.process_data(keywords, labels, country)

    def get_label_to_index(self, labels: Sequence[str]) -> dict[str, int]:
        label_to_index = {}
        for index, category in enumerate(labels):
            label_to_index[category] = index
        return label_to_index

    def get_index_to_label(self, labels: Sequence[str]) -> dict[int, str]:
        index_to_label = {}
        for index, category in enumerate(labels):
            index_to_label[index] = category
        return index_to_label

    def process_data(self, keywords: list[str], labels: list[str], country: str) -> list[tuple[list[str], str]]:
        # 遍历 dataframe
        data_list = []
        for index, keyword in enumerate(keywords):
            # 通过 index 获取 dataframe 的行
            category = labels[index]
            if not isinstance(keyword, str) or not isinstance(category, str):
                continue

            token_list = token_dict.get(country, tokenize_sg)(keyword)
            if token_list:
                data_list.append((token_list, self.label2index[category]))

        return data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[str], str]:
        return self.data[idx]


def build_vocab(dataset: KeywordCategoriesDataset):
    """
    [
        ['sharp', 'microwave', 'oven'],
        ['xiaomi', 'x10'],
    ]
    """
    documents = [text for row in dataset for text in row[0] ]
    vocab = Counter(documents)
    vocab = {token: index + 2 for index,
             (token, _) in enumerate(vocab.items())}
    vocab['<PAD>'] = 0  # type: ignore
    vocab['<UNK>'] = 1  # type: ignore

    return vocab


def collate_batch(batch, vocab: dict[str, int]):
    text_list = list()
    labels = list()
    # 每次读取一组数据
    for text, label in batch:
        text_tokens = [vocab.get(token, vocab["<UNK>"])
                       for token in text]
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_list.append(text_tensor)
        labels.append(torch.tensor(label, dtype=torch.long))

    padding_idx = vocab["<pad>"]
    # 将batch填充为相同长度文本
    text_padded = pad_sequence(
        text_list, batch_first=True, padding_value=padding_idx)
    # print("text_padded:", text_padded.shape)
    labels_tensor = torch.stack(labels)

    # 返回文本和标签的张量形式，用于后续的模型训练
    return text_padded, labels_tensor


def get_data(file_path: str, sheet_name: str) -> DataFrame:
    pkl_path = f'./data/{sheet_name}.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data

    data = pd.read_excel(file_path, sheet_name=sheet_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == '__main__':
    import pandas as pd
    data = get_data('./data/Keyword Categorization.xlsx', 'TW')
    keywords = data['Keyword'].tolist()
    labels = data['Category'].tolist()
    dataset = KeywordCategoriesDataset(keywords, labels, 'TW')
    vocab = build_vocab(dataset)
    pass
