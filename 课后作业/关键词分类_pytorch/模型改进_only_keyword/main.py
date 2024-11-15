from pandas import DataFrame
from dataset import get_df_from_csv, get_labels
from train import train
import time
from shutdown import shutdown

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


def custom_filter(data: DataFrame, category_name: str, country: str) -> DataFrame:
    if country == 'SG':
        return data[data[category_name].isin(get_labels(country))]

    return data


data_list: list[dict] = []
start_time = time.time()
for info in countries_info:
    country = info["country"]
    # if country != "TH":
    #     continue
    df = get_df_from_csv(
        f"./data/csv/{country.lower()}.csv", use_cache=True)
    keyname = 'Keyword'
    category_name = 'Category'
    sub_category_name = 'imp_level1_category_1d'
    data = df.dropna(subset=[keyname, category_name, sub_category_name]).drop_duplicates(
        subset=[keyname], keep='first').reset_index(drop=True)  # type: ignore
    # data = custom_filter(data, category_name, country)
    X = data[keyname]
    y = data[category_name]
    sub_y = data[sub_category_name]
    train(X, sub_y, y, country)
    shutdown(10)
