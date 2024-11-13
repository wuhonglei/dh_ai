import torch

from sklearn.model_selection import train_test_split
from dataset import get_data, get_df_from_csv, collate_batch, get_vocab
from train import evaluate
from dataset import KeywordCategoriesDataset, get_labels
from models.lstm_model import KeywordCategoryModel
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
data_list: list[dict] = []
for info in countries_info:
    country = info["country"]
    if country != "SG":
        continue
    df = get_df_from_csv(f"./data/csv/sg.csv", use_cache=True)
    keyname = 'Keyword'
    category_name = 'Category'
    data = df.dropna(subset=[keyname, category_name]).drop_duplicates(
        subset=[keyname], keep='first').reset_index(drop=True)  # type: ignore
    data = data[data[category_name].isin(get_labels(country))]

    X = data[keyname]
    y = data[category_name]

    dataset = KeywordCategoriesDataset(
        X.tolist(), y.tolist(), country, use_cache=True)
    # 使用 train_test_split 将数据划分为训练集和测试集
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.05, random_state=42)

    vocab = get_vocab(train_dataset, country, use_cache=True)

    # 回调函数，用于不同长度的文本进行填充
    def collate(batch): return collate_batch(batch, vocab)

    dataloader = DataLoader(dataset,
                            batch_size=128,
                            shuffle=False, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate
                                 )

    train_args = load_training_json(f"./config/{country}_params.json")
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    # 定义模型
    model = KeywordCategoryModel(
        train_args['vocab_size'], train_args['embed_dim'], train_args['hidden_size'], train_args['num_classes'], train_args['padding_idx'])
    model.load_state_dict(torch.load(
        f"./models/weights/SG_LSTM_128*2_fc_2_shopee_title_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    acc = evaluate(test_dataloader, model)
    print(f"Accuracy: {acc}")
