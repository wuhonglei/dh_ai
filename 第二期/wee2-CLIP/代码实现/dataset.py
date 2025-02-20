import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from torchvision import transforms
import pandas as pd
from PIL import Image
from typing import Literal
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence


class CLIPDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, tokenizer: DistilBertTokenizer, max_length: int, transforms: A.Compose):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transforms = transforms
        self.data = self.load_data()

    def load_data(self):
        return pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def process_image(self, image_path: str):
        image = Image.open(f"{self.image_dir}/{image_path}")
        image = np.array(image)
        transformed = self.transforms(image=image)
        return transformed['image']

    def process_caption(self, caption: str):
        encoded = self.tokenizer(
            caption,
            return_tensors='pt',
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_token_type_ids=False,)  # type: ignore
        return encoded

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        caption = self.data.iloc[idx]['caption']

        image = self.process_image(image_name)
        text = self.process_caption(caption)

        item = {
            'image': image,
            'caption': caption,
            'input_ids': text['input_ids'],
            'attention_mask': text['attention_mask'],
        }
        return item


def get_transforms(mode: Literal['train', 'test'], image_size: int):
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    input_ids = pad_sequence([item['input_ids'][0]
                             for item in batch], batch_first=True)
    attention_mask = pad_sequence(
        [item['attention_mask'][0] for item in batch], batch_first=True)

    item = {
        'image': images,
        'caption': captions,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    return item
