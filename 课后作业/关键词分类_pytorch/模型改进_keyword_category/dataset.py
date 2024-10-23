import os

from numpy import ndarray
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

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
    def __init__(self, data: DataFrame, labels, country: str, use_cache=False) -> None:
        labels = labels.tolist()
        unique_labels = list(set(labels))
        self.label2index = self.get_label_to_index(unique_labels)
        self.index2label = self.get_index_to_label(unique_labels)

        keyword_name = 'Keyword'
        sub_categories = data.drop(
            keyword_name, axis=1).to_numpy(dtype=float)
        self.data = self.process_data(
            data[keyword_name].tolist(), sub_categories, labels, country, use_cache)

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

    def process_data(self, keywords: list[str], sub_categories: ndarray, labels: list[str], country: str, use_cache: bool):
        count = len(keywords)
        cache_path = f'./tokennizer/cache/{country}_{count}.pkl'
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data

        # 遍历 dataframe
        data_list: list[tuple[list[str], list[float], int]] = []
        for index, keyword in enumerate(keywords):
            # 通过 index 获取 dataframe 的行
            category = labels[index]
            sub_category: list[float] = sub_categories[index].tolist()
            if not isinstance(keyword, str) or not isinstance(category, str):
                continue

            token_list = token_dict.get(country, tokenize_sg)(keyword.lower())
            if token_list:
                data_list.append(
                    (token_list, sub_category, self.label2index[category]))

        with open(cache_path, 'wb') as f:
            pickle.dump(data_list, f)

        return data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        title, sub_categories, label = self.data[idx]
        return title, sub_categories, label


def build_vocab(dataset: KeywordCategoriesDataset):
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
    for (word, freq) in vocab.items():
        if freq >= 1:
            word_2_index[word] = len(word_2_index)

    return word_2_index


def get_vocab(train_dataset: KeywordCategoriesDataset, file_path: str, use_cache: bool = True):

    if use_cache and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    vocab = build_vocab(train_dataset)
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)
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
    pkl_path = f'./data/excel.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data[sheet_name] if sheet_name else data

    data = pd.read_excel(file_path, sheet_name=None, dtype=str)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    return data[sheet_name] if sheet_name else data


def get_df_from_csv(file_path: str, use_cache=True) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    filename = file_path.split('/')[-1].split('.')[0]
    cache_name = f'./data/cache/{filename}_csv.pkl'
    if use_cache and os.path.exists(cache_name):
        with open(cache_name, 'rb') as f:
            df = pickle.load(f)
        return df

    with open(cache_name, 'wb') as f:
        pickle.dump(df, f)

    return df


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
