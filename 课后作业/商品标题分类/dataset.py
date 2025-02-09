import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class TitleDataset(Dataset):
    def __init__(self, data_path: str, title_name: str, label_names: list[str]):
        self.data_path = data_path
        self.title_name = title_name
        self.label_names = label_names
        self.data = self.load_data()

    def load_data(self):
        return pd.read_csv(self.data_path,
                           usecols=[self.title_name, *self.label_names], dtype={
                               self.title_name: str,
                               **{label_name: 'category' for label_name in self.label_names}
                           })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        title = item[self.title_name]
        for label_name in self.label_names:
            label = item[label_name]
            if isinstance(label, str):
                break

        return title, label


def collate_fn(batch: list[tuple[str, str]], label_encoder: LabelEncoder, tokenizer):
    titles = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Tokenize titles using BERT tokenizer
    encoded_titles = tokenizer(
        titles,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )
    labels = label_encoder.transform(labels)
    labels = torch.tensor(labels, dtype=torch.long)
    return encoded_titles['input_ids'], encoded_titles['attention_mask'], labels
