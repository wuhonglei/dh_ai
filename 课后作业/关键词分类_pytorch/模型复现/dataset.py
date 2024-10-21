from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
import re
import nltk
import string


def get_stop_words(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return file.read().split()


class KeywordCategoriesDataset(Dataset):
    def __init__(self, data: DataFrame, language: str) -> None:
        stop_words = set(stopwords.words(
            language) + get_stop_words('./stopwords_custom.txt'))

        # 遍历 dataframe
        data_list = []
        for index, row in data.iterrows():
            # 通过 index 获取 dataframe 的行
            keyword = row["Keyword"]
            category = row["Category"]
            if not isinstance(keyword, str) or not isinstance(category, str):
                continue

            keyword = re.sub(r'[^\w\s]', '', keyword)
            token_list = nltk.word_tokenize(keyword)
            new_token_list = []
            for token in token_list:
                strip_token = token.strip()
                if token not in stop_words and len(strip_token) > 1:
                    new_token_list.append(strip_token)

            if len(new_token_list) > 0:
                data_list.append((' '.join(new_token_list), category))

        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_excel('./data/Keyword Categorization.xlsx', sheet_name=None)
    dataset = KeywordCategoriesDataset(data['SG'], 'english')
