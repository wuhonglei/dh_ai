"""
数据集
"""

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer

from config import test_csv_path, columns


class BaseDataset(Dataset):
    def __init__(self, csv_path: str, column_name: str, level1_label_name: str, leaf_label_name: str, tokenizer, max_length: int):
        self.data = pd.read_csv(csv_path).dropna(
            subset=[column_name, level1_label_name])
        self.column_name = column_name
        self.level1_label_name = level1_label_name
        self.leaf_label_name = leaf_label_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.level1_label_encoder = LabelEncoder()
        self.level1_label_encoder.fit(self.data[self.level1_label_name])
        self.leaf_label_encoder = LabelEncoder()
        self.leaf_label_encoder.fit(self.data[self.leaf_label_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row[self.column_name]
        level1_label = row[self.level1_label_name]
        level1_label_id = self.level1_label_encoder.transform(
            [level1_label]).item()  # type: ignore
        leaf_label = row[self.leaf_label_name]
        leaf_label_id = self.leaf_label_encoder.transform(
            [leaf_label]).item()  # type: ignore

        # 对文本进行token化
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'level1': torch.tensor(level1_label_id, dtype=torch.long),
            'leaf': torch.tensor(leaf_label_id, dtype=torch.long)
        }


if __name__ == '__main__':
    level1_label_name = 'global_be_category_id'
    leaf_label_name = 'global_be_category_id_level1'
    max_length = 28
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = BaseDataset(
        csv_path=test_csv_path,
        column_name=columns[0],
        level1_label_name=level1_label_name,
        leaf_label_name=leaf_label_name,
        tokenizer=tokenizer,
        max_length=max_length)
    print(len(dataset))
    print(dataset[0])
