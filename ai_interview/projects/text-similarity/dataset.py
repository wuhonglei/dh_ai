from torch.utils.data import Dataset
from memory_profiler import profile
import pandas as pd
import json


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

    def __getitem__(self, index):
        item = self.data.iloc[index]
        return item['title'], item['content']


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

    def __getitem__(self, index):
        item = self.data[index]
        return item['title'], item['content']


if __name__ == "__main__":
    dataset = NewsDatasetJson("./data/origin/sohu_data.json")
    print(len(dataset))
    print(dataset[0])
