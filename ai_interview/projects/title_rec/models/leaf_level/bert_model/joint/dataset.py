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
    def __init__(self, csv_path: str, column_name: str, label_name: str, tokenizer, max_length: int):
        self.data = pd.read_csv(csv_path).dropna(
            subset=[column_name, label_name])
        self.column_name = column_name
        self.label_name = label_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data[self.label_name])

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
    label_name = 'global_be_category_id'
    max_length = 28
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = BaseDataset(
        csv_path=test_csv_path,
        column_name=columns[0],
        label_name=label_name,
        tokenizer=tokenizer,
        max_length=max_length)
    print(len(dataset))
    print(dataset[0])
