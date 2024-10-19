import pandas as pd
from dataset import KeywordCategoriesDataset
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
excel = pd.read_excel("./data/Keyword_Categorization.xlsx",
                      sheet_name=None, dtype=str)
for info in countries_info:
    sheet_name = info["country"]
    stopwords = info["stopwords"]
    dataset = KeywordCategoriesDataset(
        excel[sheet_name], stopwords)  # type: ignore
    dataloader = DataLoader(dataset, batch_size=len(dataset))

    for keywords_list, category_list in dataloader:
        vectorizer = CountVectorizer()
        tf_keywords = vectorizer.fit_transform(keywords_list)
        x_train, x_test, y_train, y_test = train_test_split(
            tf_keywords.toarray(), category_list, random_state=0, test_size=0.2
        )

        model = LinearSVC()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        result = {
            "Country": sheet_name,
            "Accuracy": str(int(accuracy_score(y_test, y_pred) * 100)) + "%",
            "Precision": str(int(precision_score(y_test, y_pred, average="weighted") * 100))
            + "%",
            "Recall": str(int(recall_score(y_test, y_pred, average="weighted") * 100))
            + "%",
            "F1": str(int(f1_score(y_test, y_pred, average="weighted") * 100)) + "%",
        }
        print(result)
        data_list.append(result)

df = pd.DataFrame(data_list)
df.to_csv("./csv/模型复现.csv", index=False)
