import os
import json

from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split

from tokennizer.sg import tokenize_sg
from tokennizer.my import tokenize_my
from tokennizer.th import tokenize_th
from tokennizer.tw import tokenize_tw

from utils.common import exists_cache, save_cache, load_cache, get_file_state, calculate_md5, save_json, load_json

from typing import Sequence

token_dict = {
    'SG': tokenize_sg,
    'MY': tokenize_my,
    'TH': tokenize_th,
    'TW': tokenize_tw,
}


class KeywordCategoriesDataset(Dataset):
    def __init__(self, keywords: list[str], labels: list[str], country: str, use_cache=False) -> None:
        unique_labels = get_labels(country)
        self.label2index = self.get_label_to_index(unique_labels)
        self.index2label = self.get_index_to_label(unique_labels)
        self.data = self.process_data(keywords, labels, country, use_cache)

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

    def process_data(self, keywords: list[str], labels: list[str], country: str, use_cache: bool) -> list[tuple[list[str], str]]:
        count = len(keywords)
        cache_path = f'./cache/tokennizer/{country}_{count}_{calculate_md5("".join(keywords[0:10]))}.pkl'
        if use_cache and exists_cache(cache_path):
            data = load_cache(cache_path)
            return data

        # 遍历 dataframe
        data_list = []
        for index, keyword in enumerate(keywords):
            # 通过 index 获取 dataframe 的行
            category = labels[index]
            if not isinstance(keyword, str) or not isinstance(category, str):
                continue

            token_list = token_dict.get(country, tokenize_sg)(keyword.lower())
            if token_list:
                data_list.append((token_list, self.label2index[category]))

        save_cache(cache_path, data_list)
        return data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[str], str]:
        return self.data[idx]


def build_vocab(dataset: KeywordCategoriesDataset, min_freq: int = 10):
    """
    [
        ['sharp', 'microwave', 'oven'],
        ['xiaomi', 'x10'],
    ]
    """
    documents = [text for row in dataset for text in row[0]]
    vocab = Counter(documents)
    word_2_index = {
        '<PAD>': 0,
        '<UNK>': 1,
    }
    vocab_list = []
    for word, freq in vocab.items():
        vocab_list.append((word, freq))
    vocab_list.sort(key=lambda x: x[1], reverse=True)
    for (word, freq) in vocab_list:
        if freq >= min_freq:
            word_2_index[word] = len(word_2_index)
    return word_2_index


def get_vocab(train_dataset: KeywordCategoriesDataset, country: str, use_cache: bool = True):
    seq_list = [''.join(row[0]) for row in train_dataset[0:10]]  # type: ignore
    # cache_name = f"./cache/vocab/{country}_vocab_{len(train_dataset)}_{calculate_md5(''.join(seq_list))}.json"
    cache_name = './cache/vocab/SG_vocab_827233_cd306371b43b970ea53736ed2589778b.json'
    if use_cache and exists_cache(cache_name):
        vocab = load_json(cache_name)
        return vocab

    vocab = build_vocab(train_dataset, 1000)
    save_json(cache_name, vocab)
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

    padding_idx = vocab['<PAD>']
    # 将batch填充为相同长度文本
    text_padded = pad_sequence(
        text_list, batch_first=True, padding_value=padding_idx)
    labels_tensor = torch.stack(labels)

    # 返回文本和标签的张量形式，用于后续的模型训练
    return text_padded, labels_tensor


def get_data(file_path: str, sheet_name: str = ''):
    cache_name = os.path.abspath(f'./cache/data/excel.pkl')
    if exists_cache(cache_name):
        data = load_cache(cache_name)
        return data[sheet_name] if sheet_name else data

    data = pd.read_excel(file_path, sheet_name=None, dtype=str)
    save_cache(cache_name, data)
    return data[sheet_name] if sheet_name else data


def get_df_from_csv(file_path: str, use_cache=True) -> pd.DataFrame:
    filename = file_path.split('/')[-1].split('.')[0]
    stat = get_file_state(file_path)
    # type: ignore
    cache_name = f'./cache/data/{filename}_{stat.st_size}_{int(stat.st_mtime)}_csv.pkl'
    if use_cache and exists_cache(cache_name):
        df = load_cache(cache_name)
        return df

    df = pd.read_csv(file_path)
    save_cache(cache_name, df)
    return df


def get_labels(country: str) -> list[str]:
    with open(f'./config/{country}_label_to_index.json', 'r') as f:
        label_to_index = json.load(f)
    return list(label_to_index.keys())


if __name__ == '__main__':
    import pandas as pd
    country = 'SG'
    excel = get_data('./data/Keyword Categorization.xlsx')
    data = excel[country].drop_duplicates(
        subset=['Keyword'], keep='first').reset_index(drop=True)  # type: ignore
    X = data["Keyword"]
    y = data["Category"]

    # 使用 train_test_split 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), y.tolist(), test_size=0.05, random_state=0)

    train_dataset = KeywordCategoriesDataset(
        X_train, y_train, country)
    test_dataset = KeywordCategoriesDataset(
        X_test, y_test, country)

    train_vocab = build_vocab(train_dataset)
    test_vocab = build_vocab(test_dataset)

    train_vocab_words = set(train_vocab.keys())
    test_vocab_words = set(test_vocab.keys())

    with open(f"./vocab/{country}/train_vocab.txt", "w") as f:
        f.write('\n'.join(train_vocab_words))

    with open(f"./vocab/{country}/test_vocab.txt", "w") as f:
        f.write('\n'.join(test_vocab_words))

    def read_vocab(vocab_path):
        with open(vocab_path, 'r') as f:
            return f.readlines()

    def write_vocab(vocab_path, words):
        with open(vocab_path, 'w') as f:
            f.writelines(words)

    # 分析训练集和测试集的分词结果的差异
    train_vocab_path = f'./vocab/{country}/train_vocab.txt'
    test_vocab_path = f'./vocab/{country}/test_vocab.txt'

    train_words = read_vocab(train_vocab_path)
    test_words = read_vocab(test_vocab_path)
    train_words, test_words = set(train_words), set(test_words)
    print(f'训练集词汇量: {len(train_words)}')
    print(f'测试集词汇量: {len(test_words)}')
    print(f'测试集和训练集词汇量差异: {len(test_words - train_words)}')
    print(f'测试集和训练集词汇量差异: {(test_words - train_words)}')
    write_vocab(f'./vocab/{country}/test_train_diff_vocab.txt',
                test_words - train_words)
