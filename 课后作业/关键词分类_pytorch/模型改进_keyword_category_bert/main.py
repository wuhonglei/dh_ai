from sklearn.model_selection import train_test_split
from dataset import get_data, get_df_from_csv
from train import train
import time

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
    df = get_df_from_csv(f"./data/csv/{country.lower()}.csv")
    data = df.drop_duplicates(
        # type: ignore
        subset=['Keyword'], keep='first').reset_index(drop=True)
    X = data.drop(columns=["Category"], axis=1)
    y = data["Category"]

    train(X, y, country)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
