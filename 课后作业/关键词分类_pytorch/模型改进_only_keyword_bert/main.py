from sklearn.model_selection import train_test_split
from dataset import get_data, get_df_from_csv
from train import train
import pandas as pd
import time
import random

countries_info = [
    {"country": "SG", "stopwords": "english"},
    {"country": "MY", "stopwords": "english"},
    {"country": "TH", "stopwords": "english"},
    {"country": "TW", "stopwords": "chinese"},
    {"country": "VN", "stopwords": "english"},
    {"country": "ID", "stopwords": "english"},
    {"country": "PH", "stopwords": "english"},
    {"country": "BR", "stopwords": "spanish"},
    {"country": "MX", "stopwords": "spanish"},
    {"country": "CL", "stopwords": "spanish"},
    {"country": "CO", "stopwords": "spanish"},
]

data_list: list[dict] = []
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue

    df_sg = get_df_from_csv(f"./data/sg.csv", use_cache=False)
    df_other = get_df_from_csv(
        f"./data/sg_final_without_sg.csv", use_cache=False)

    # 测试集
    rows_index = random.choices(range(len(df_sg)), k=2000)
    df_sg_test = df_sg.iloc[rows_index]
    df_sg_train = df_sg.drop(rows_index)

    df_train = pd.concat([df_sg_train, df_other], ignore_index=True, axis=0)

    # X = data["Keyword"]
    # y = data["sg_category"]

    train(df_train, df_sg_test, country)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
