import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import get_data
from train import train

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
excel = get_data('./data/Keyword Categorization.xlsx')
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue
    data = excel[country].drop_duplicates(
        subset=['Keyword'], keep='first').reset_index(drop=True)  # type: ignore
    X = data["Keyword"]
    y = data["Category"]
    # 使用 train_test_split 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), y.tolist(), test_size=0.2, random_state=42)

    train(X_train, y_train, country, X_test, y_test)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
