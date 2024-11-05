import torch

from sklearn.model_selection import train_test_split
from dataset import get_data, get_df_from_csv
from train import evaluate
import time
from dataset import KeywordCategoriesDataset
from models.simple_model import KeywordCategoryModel
from torch.utils.data import DataLoader
from utils.model import load_training_json

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

bert_name = 'bert-base-uncased'
ignored_country_categories = {
    'SG': [
        'Lifestyle',  # 2.06%
        'Others',  # 1.32%
        'Platform Terms (Lazada / Shopee)',  # 1.05%
        'Miscellaneous (Adult)',  # 0.01%
        'Pets & Supplies'  # 0.01%
    ]
}

data_list: list[dict] = []
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue
    df = get_df_from_csv(f"./data/csv/sg.csv", use_cache=True)
    data = df.drop_duplicates(
        subset=['Keyword'], keep='first').reset_index(drop=True)  # type: ignore

    category_name = 'Category'
    data = data[~df[category_name].isin(
        ignored_country_categories.get(country, []))]
    X = data["Keyword"]
    y = data[category_name]

    dataset = KeywordCategoriesDataset(bert_name,
                                       X.tolist(), y.tolist(), country, use_cache=True)
    # 使用 train_test_split 将数据划分为训练集和测试集
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.05, random_state=42)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)

    train_args = load_training_json(f"./config/{country}_params.json")
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    # 定义模型
    model = KeywordCategoryModel(
        bert_name, train_args['hidden_size'], train_args[' num_classes'], train_args['dropout'])
    model.to(DEVICE)
    model.load_state_dict(torch.load(
        f"./models/weights/{country}_model.pth", map_location=DEVICE, weights_only=True))

    acc = evaluate(test_dataloader, model)
    print(f"Accuracy: {acc}")


# df = pd.DataFrame(data_list)
# df.to_csv("./csv/模型复现.csv", index=False)
