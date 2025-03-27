import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

from utils.common import exists_cache, load_cache, save_cache


class TitleDataset(Dataset):
    def __init__(self, data_path: str, title_name: str, label_names: list[str], tokenizer: BertTokenizer, cache_name: str):
        self.data_path = data_path
        self.title_name = title_name
        self.label_names = label_names
        self.tokenizer = tokenizer
        self.data = self.load_data()
        cache_path = os.path.join('cache', cache_name)
        if exists_cache(cache_path):
            self.encoded_titles = load_cache(cache_path)
        else:
            # 预先对所有标题进行tokenization
            self.encoded_titles = self.tokenize_titles()
            save_cache(cache_path, self.encoded_titles)

    def load_data(self):
        return pd.read_csv(self.data_path,
                           usecols=[self.title_name, *self.label_names], dtype={
                               self.title_name: str,
                               **{label_name: 'category' for label_name in self.label_names}
                           })

    def tokenize_titles(self):
        titles = self.data[self.title_name].tolist()
        # 单独对每个标题进行tokenization
        encoded_list = [self.tokenizer(
            title,
            padding=True,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            return_token_type_ids=False,
        ) for title in titles]

        return encoded_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # 返回预先计算好的tokens而不是原始文本
        encoded_title = {
            'input_ids': self.encoded_titles[idx]['input_ids'][0],
            'attention_mask': self.encoded_titles[idx]['attention_mask'][0]
        }
        for label_name in self.label_names:
            label = item[label_name]
            if isinstance(label, str):
                break
        return encoded_title, label


def collate_fn(batch: list[tuple[dict, str]], label_encoder: LabelEncoder):
    encoded_titles = {
        # 对齐长度, 不足的用0填充 pad_sequence
        'input_ids': pad_sequence([item[0]['input_ids'] for item in batch], batch_first=True),
        'attention_mask': pad_sequence([item[0]['attention_mask'] for item in batch], batch_first=True)
    }
    labels = [item[1] for item in batch]
    labels = label_encoder.transform(labels)
    labels = torch.tensor(labels, dtype=torch.long)
    return encoded_titles['input_ids'], encoded_titles['attention_mask'], labels
