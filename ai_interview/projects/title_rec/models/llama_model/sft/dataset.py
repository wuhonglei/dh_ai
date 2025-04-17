from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import pandas as pd


class TitleRecDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, name_col: str, label_id_col: str, id_to_name: dict[int, str], categories: list[str]):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.name_col = name_col
        self.label_id_col = label_id_col
        self.id_to_name = id_to_name
        self.categories = categories

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row[self.name_col]
        label_id = row[self.label_id_col]
        label_name = self.id_to_name[label_id]
        max_length = 128
        category_token_len = 80
        name_token_limit = max_length - category_token_len
        name_token_list = name.split()
        if len(name_token_list) > name_token_limit:
            name_token_list = name_token_list[:name_token_limit]
        name = ' '.join(name_token_list)
        prompt = f'Text: "{name}"\nWhich category? [{", ".join(self.categories)}]\nAnswer:'
        return {
            "input_ids": self.tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            "labels": self.tokenizer(label_name, truncation=True, padding="max_length", max_length=max_length)["input_ids"]
        }
