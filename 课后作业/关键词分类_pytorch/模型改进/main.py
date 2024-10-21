import pandas as pd
from dataset import KeywordCategoriesDataset
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import train, evaluate

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
excel = pd.read_excel("./data/Keyword Categorization.xlsx",
                      sheet_name=None, dtype=str)
for info in countries_info:
    country = info["country"]
    if country != "TW":
        continue

    X = excel[country]["Keyword"]
    y = excel[country]["Category"]
    # 使用 train_test_split 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), y.tolist(), test_size=0.2, random_state=42)

    train(X_train, y_train, country, X_test, y_test)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
