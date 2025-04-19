import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from typing import Optional, TypedDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch


class MultiModalData(TypedDict):
    image: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class MultiModalDataset(Dataset):
    def __init__(self, csv_path: str, title_col_name: str, image_col_name: str, label_col_name: str, tokenizer, max_length: int, label_encoder: Optional[LabelEncoder], image_dir: str, transform: A.Compose):
        self.data = pd.read_csv(csv_path).dropna(
            subset=[title_col_name, image_col_name, label_col_name])
        self.title_col_name = title_col_name
        self.image_col_name = image_col_name
        self.label_col_name = label_col_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> MultiModalData:
        row = self.data.iloc[index]
        title = row[self.title_col_name]
        image_name = row[self.image_col_name]
        label = row[self.label_col_name]
        label_id = self.label_encoder.transform([label]).item()  # type: ignore

        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']

        encoding = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
