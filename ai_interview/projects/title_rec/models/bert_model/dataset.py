"""
数据集
"""

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer
from typing import Optional

from config import train_csv_path, test_csv_path, columns, label_name, max_length


class BaseDataset(Dataset):
    def __init__(self, csv_path: str, column_name: str, label_name: str, tokenizer, max_length: int, label_encoder: Optional[LabelEncoder]):
        self.data = pd.read_csv(csv_path).dropna(
            subset=[column_name, label_name])
        self.column_name = column_name
        self.label_name = label_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row[self.column_name]
        label = row[self.label_name]
        label_id = self.label_encoder.transform([label]).item()  # type: ignore

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
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()
    # 先拟合训练集的标签
    train_data = pd.read_csv(train_csv_path)
    label_encoder.fit(train_data[label_name])

    # 使用同一个label_encoder创建数据集
    train_dataset = BaseDataset(
        csv_path=train_csv_path,
        column_name=columns[0],
        label_name=label_name,
        tokenizer=tokenizer,
        max_length=max_length,
        label_encoder=label_encoder
    )

    test_dataset = BaseDataset(
        csv_path=test_csv_path,
        column_name=columns[0],
        label_name=label_name,
        tokenizer=tokenizer,
        max_length=max_length,
        label_encoder=label_encoder
    )
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset[0])
    print(test_dataset[0])
