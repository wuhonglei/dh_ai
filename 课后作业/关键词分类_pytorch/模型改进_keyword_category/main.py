from sklearn.model_selection import train_test_split
from dataset import get_data, get_df_from_csv
# from train import train
from train import train
import time
import pandas as pd

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
start_time = time.time()
print(f"Time taken to load data: {time.time() - start_time:.2f}s")
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue
    df = get_df_from_csv(f"./data/csv/{country}.csv")
    data = df.drop_duplicates(
        subset=['Keyword'], keep='first').reset_index(drop=True)  # type: ignore

    dummy_cols = ['imp_level1_category_1d', 'pv_level1_category_1d',
                  'order_level1_category_1d']
    one_hot_df = pd.get_dummies(data[dummy_cols], columns=dummy_cols)
    X = pd.concat([data['Keyword'], one_hot_df], axis=1)
    y = data["Category"]
    # 使用 train_test_split 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=0)

    train(X_train, y_train, country, X_test, y_test)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
