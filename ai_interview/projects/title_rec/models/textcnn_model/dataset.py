"""
数据集类，用于加载和处理数据
如果 word_to_id 为 None，则使用 tokenizer 将文本转换为 tokens
如果 word_to_id 不为 None，则使用 word_to_id 将 tokens 转换为 token_ids
如果 max_length 为 None，则不进行截断
"""

from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Callable
from sklearn.preprocessing import LabelEncoder


class TextCNNDataset(Dataset):
    def __init__(self, csv_path: str, column_name: str, label_name: str, tokenizer: Callable, word_to_id: dict[str, int], max_length: int):
        self.column_name = column_name
        self.label_name = label_name
        self.data = pd.read_csv(csv_path).dropna(
            subset=[column_name, label_name])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data[self.label_name])
        self.tokenizer = tokenizer
        self.word_to_id = word_to_id
        self.max_length = max_length
        self.pad_id = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = self.tokenizer(row[self.column_name])
        label = row[self.label_name]
        if self.max_length and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        if self.word_to_id:
            token_ids = [self.word_to_id.get(token, self.pad_id)
                         for token in tokens]
            if len(token_ids) < self.max_length:
                token_ids += [self.pad_id] * (self.max_length - len(token_ids))
            labels_idx = self.label_encoder.transform(
                [label]).item()  # type: ignore
            return torch.tensor(token_ids, dtype=torch.long), torch.tensor(labels_idx, dtype=torch.long)
        else:
            return tokens, label
