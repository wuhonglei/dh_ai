from typing import Sequence
import os

import numpy as np
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

# from tokennizer.sg import tokenize_sg
# from tokennizer.my import tokenize_my
# from tokennizer.th import tokenize_th
# from tokennizer.tw import tokenize_tw

from utils.common import exists_cache, save_cache, load_cache

from transformers import BertTokenizer
import torch


# 分词与编码函数


# token_dict = {
#     'SG': tokenize_sg,
#     'MY': tokenize_my,
#     'TH': tokenize_th,
#     'TW': tokenize_tw,
# }


class KeywordCategoriesDataset(Dataset):
    def __init__(self, model_name: str, keywords: list[str], labels: list[str], country: str, use_cache=False) -> None:
        # 2. 标签编码
        label_encoder = LabelEncoder()
        self.label_encoder = label_encoder
        self.labels: list[int] = label_encoder.fit_transform(
            labels).tolist()  # type: ignore

        encodings_cache_name = os.path.abspath(
            f'./cache/dataset/{country}_{len(keywords)}_{keywords[0].split()[0]}_encodings.pkl')
        if use_cache and exists_cache(encodings_cache_name):
            encodings = load_cache(encodings_cache_name)
        else:
            # 加载预训练的BERT分词器
            # tokenizer = BertTokenizer.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            seq_lengths = [len(keyword.split()) for keyword in keywords]
            # MAX_LEN = int(np.percentile(seq_lengths, 95))  # 选择95%分位数作为最大长度
            MAX_LEN = 6  # 选择95%分位数作为最大长度
            encodings = self.encode_texts(keywords, tokenizer, MAX_LEN)
            save_cache(encodings_cache_name, encodings)

        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']

    def get_max_len(self, data: DataFrame) -> int:
        """
        获取数据集中文本的最大长度
        """
        max_len = 0
        for text in data['text']:
            max_len = max(max_len, len(text))
        return max_len

    def encode_texts(self, texts, tokenizer, max_len):
        return tokenizer(
            texts,
            add_special_tokens=True,      # 添加[CLS]和[SEP]
            max_length=max_len,
            padding='max_length',         # 填充到最大长度
            truncation=True,              # 超出长度则截断
            return_attention_mask=True,
            return_token_type_ids=False,  # 对于单句分类任务不需要
            return_tensors='pt'           # 返回PyTorch张量
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.attention_masks[idx], torch.tensor(self.labels[idx])


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
    cache_name = os.path.abspath(f'./cache/data/excel.pkl')
    if exists_cache(cache_name):
        data = load_cache(cache_name)
        return data[sheet_name] if sheet_name else data

    data = pd.read_excel(file_path, sheet_name=None, dtype=str)
    save_cache(cache_name, data)
    return data[sheet_name] if sheet_name else data


def get_df_from_csv(file_path: str, use_cache=True) -> pd.DataFrame:
    filename = file_path.split('/')[-1].split('.')[0]
    cache_name = os.path.abspath(f'./cache/data/{filename}_csv.pkl')
    if use_cache and exists_cache(cache_name):
        df = load_cache(cache_name)
        return df

    df = pd.read_csv(file_path)
    save_cache(cache_name, df)
    return df
