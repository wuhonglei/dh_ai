from sklearn.model_selection import train_test_split
from dataset import get_data, get_df_from_csv, get_labels
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
start_time = time.time()
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue
    df = get_df_from_csv(
        f"./data/csv/{country.lower()}.csv", use_cache=True)
    keyname = 'Keyword'
    category_name = 'Category'
    data = df.dropna(subset=[keyname, category_name]).drop_duplicates(
        subset=[keyname], keep='first').reset_index(drop=True)  # type: ignore
    data = data[data[category_name].isin(get_labels(country))]
    X = data[keyname]
    y = data[category_name]
    train(X, y, country)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
