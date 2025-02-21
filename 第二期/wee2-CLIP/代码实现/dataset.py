import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from torchvision import transforms
import pandas as pd
from PIL import Image
from typing import Literal, List
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence


class CLIPDataset(Dataset):
    def __init__(self, image_filenames: List[str] | None, captions: List[str] | None, tokenizer: DistilBertTokenizer | None, max_length: int | None, transforms: A.Compose | None):
        self.image_filenames = image_filenames
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transforms = transforms

    def __len__(self):
        if self.image_filenames:
            return len(self.image_filenames)
        elif self.captions:
            return len(self.captions)
        else:
            return 0

    def process_image(self, image_path: str | None):
        if image_path is None:
            return None

        image = Image.open(image_path)
        image = np.array(image)
        transformed = self.transforms(image=image)  # type: ignore
        return transformed['image']

    def process_caption(self, caption: str | None):
        if caption is None:
            return None

        encoded = self.tokenizer(
            caption,
            return_tensors='pt',
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_token_type_ids=False,)  # type: ignore
        return encoded

    def __getitem__(self, idx):
        image_name = None
        caption = None
        if self.image_filenames:
            image_name = self.image_filenames[idx]

        if self.captions:
            caption = self.captions[idx]

        image = self.process_image(image_name)
        text = self.process_caption(caption)

        item = {
            'image': image,
            'caption': caption,
            'input_ids': text['input_ids'][0] if text else None,
            'attention_mask': text['attention_mask'][0] if text else None,
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
    has_image = any(item['image'] is not None for item in batch)
    has_caption = any(item['caption'] is not None for item in batch)

    if has_image:
        images = torch.stack([item['image'] for item in batch])
    else:
        images = None

    if has_caption:
        captions = [item['caption'] for item in batch]
        input_ids = pad_sequence([item['input_ids']
                                  for item in batch], batch_first=True)
        attention_mask = pad_sequence(
            [item['attention_mask'] for item in batch], batch_first=True)
    else:
        captions = None
        input_ids = None
        attention_mask = None

    item = {
        'image': images,
        'caption': captions,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    return item
