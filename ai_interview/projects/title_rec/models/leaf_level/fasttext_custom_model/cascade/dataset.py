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


class FastTextDataset(Dataset):
    def __init__(self, csv_path: str, column_name: str, level1_label_name: str, leaf_label_name: str, tokenizer: Callable, word_to_id: dict[str, int], max_length: int, wordNgrams: int, level1_label_encoder: LabelEncoder, leaf_label_encoder: LabelEncoder):
        self.column_name = column_name
        self.level1_label_name = level1_label_name
        self.leaf_label_name = leaf_label_name
        self.data = pd.read_csv(csv_path).dropna(
            subset=[column_name, level1_label_name, leaf_label_name])
        self.level1_label_encoder = level1_label_encoder
        self.leaf_label_encoder = leaf_label_encoder
        self.tokenizer = tokenizer
        self.word_to_id = word_to_id
        self.max_length = max_length
        self.pad_id = 0
        self.unk_id = 1
        self.wordNgrams = wordNgrams

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        unigrams = self.tokenizer(row[self.column_name])
        if self.wordNgrams > 1:
            ngrams = [unigrams[i] + '_' + unigrams[i + 1]
                      for i in range(len(unigrams) - self.wordNgrams + 1)]
            unigrams = unigrams + ngrams

        level1_label = row[self.level1_label_name]
        leaf_label = row[self.leaf_label_name]
        if self.max_length and len(unigrams) > self.max_length:
            unigrams = unigrams[:self.max_length]

        if self.word_to_id:
            token_ids = [self.word_to_id.get(token, self.unk_id)
                         for token in unigrams]
            if len(token_ids) < self.max_length:
                token_ids += [self.pad_id] * (self.max_length - len(token_ids))
            level1_label_idx = self.level1_label_encoder.transform(
                [level1_label]).item()  # type: ignore
            leaf_label_idx = self.leaf_label_encoder.transform(
                [leaf_label]).item()  # type: ignore
            return torch.tensor(token_ids, dtype=torch.long), torch.tensor(level1_label_idx, dtype=torch.long), torch.tensor(leaf_label_idx, dtype=torch.long)
        else:
            return unigrams, level1_label, leaf_label
