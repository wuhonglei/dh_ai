from torch.utils.data import Dataset
import pandas as pd
import json
from typing import TypedDict


class NewsItem(TypedDict):
    index: int
    title: str
    content: str


class NewsDatasetCsv(Dataset):
    """
    从csv文件中读取数据
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        return pd.read_csv(self.data_path)

    def __getitem__(self, index) -> NewsItem:
        item = self.data.iloc[index]
        return {
            'index': item['index'],
            'title': item['title'],
            'content': item['content']
        }


class NewsDatasetJson(Dataset):
    """
    从json文件中读取数据
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        return data

    def __getitem__(self, index) -> NewsItem:
        item = self.data[index]
        return {
            'index': item['index'],
            'title': item['title'],
            'content': item['content']
        }


if __name__ == "__main__":
    dataset = NewsDatasetJson("./data/origin/sohu_data.json")
    print(len(dataset))
    print(dataset[0])
