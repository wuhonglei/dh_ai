import os

import torch
import pickle
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class NewsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: BertTokenizer, use_cache=True, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        cache_name = self.get_cache_name(texts)
        if use_cache and self.has_cache(cache_name):
            self.encodings = self.load_cache(cache_name)
        else:
            self.encodings = [self.get_encoding(
                text, max_length) for text in tqdm(texts)]
            self.save_cache(cache_name, self.encodings)

    def __len__(self):
        return len(self.texts)

    def get_encoding(self, text: str, max_length: int = 512):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        encoding = self.encodings[idx]

        return {
            # 张量形状：[max_length]
            'input_ids': encoding['input_ids'].flatten(),
            # 张量形状：[max_length]
            'attention_mask': encoding['attention_mask'].flatten(),
            # 张量形状：[max_length]
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # 标签
        }

    def get_cache_name(self, texts: list[str]):
        total = sum([len(text) for text in texts])
        return f"cache/dataset_{len(texts)}_{total}.pkl"

    def has_cache(self, cache_name: str):
        try:
            with open(cache_name, "rb") as f:
                return True
        except FileNotFoundError:
            return False

    def load_cache(self, cache_name: str) -> list[BatchEncoding]:
        with open(cache_name, "rb") as f:
            return pickle.load(f)

    def save_cache(self, cache_name: str, data: list[BatchEncoding]):
        if not os.path.exists("cache"):
            os.makedirs("cache")

        with open(cache_name, "wb") as f:
            pickle.dump(data, f)
