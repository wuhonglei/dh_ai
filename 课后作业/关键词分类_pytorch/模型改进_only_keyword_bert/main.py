from sklearn.model_selection import train_test_split
from dataset import get_data
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
excel = get_data('./data/Keyword Categorization.xlsx')
print(f"Time taken to load data: {time.time() - start_time:.2f}s")
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue
    data = excel[country].drop_duplicates(
        subset=['Keyword'], keep='first').reset_index(drop=True)  # type: ignore
    X = data["Keyword"]
    y = data["Category"]

    train(X, y, country)


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
