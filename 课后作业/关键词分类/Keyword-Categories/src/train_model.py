import pathlib
import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.svm import LinearSVC

from src.utils import timer

logger = logging.getLogger(__name__)


@timer
def model_selection(
    country: str,
    process_data: pd.DataFrame,
) -> None:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    models = [
        LinearSVC,
        RandomForestClassifier,
        MultinomialNB,
        KNeighborsClassifier,
        DecisionTreeClassifier,
    ]

    param = {
        "C": [1.3, 1.5, 1.8, 2.3, 2.5, 2.8],
        # "dual": (True, False),
        "random_state": [666],
    }

    data = process_data[pd.notnull(process_data["Category"])]

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(["Category"], axis=1),
        data["Category"],
        random_state=0,
        test_size=0.2,
    )

    result = []

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV

    for model in models:

        # main = GridSearchCV(model(), param_grid, refit=True, verbose=3, n_jobs=-1)
        # main = RandomizedSearchCV(model(), param, n_iter=5, cv=10)
        main = model()
        main.fit(x_train, y_train)
        y_pred = main.predict(x_test)
        result.append(
            {
                "Country": country,
                "Model": model.__name__,
                "Accuracy": str(int(accuracy_score(y_test, y_pred) * 100)) + "%",
                "F1": str(int(f1_score(y_test, y_pred, average="weighted") * 100))
                + "%",
            },
        )

    df = pd.DataFrame(result)

    path = pathlib.Path().resolve() / "csv" / "model_selection_result.csv"
    hdr = False if path.exists() else True
    df.to_csv(
        path,
        mode="a",
        header=hdr,
        index=False,
    )
    logger.info(df)


@timer
def train_model(
    country: str,
    process_data: pd.DataFrame,
    raw_data: pd.DataFrame,
):

    data = process_data[pd.notnull(process_data["Category"])]

    # train_data = data[data["Category"].str.len() >= 1]
    # test_data = data[data["Category"].str.len() == 0]
    # raw_data = raw_data[raw_data["Category"].str.len() == 0]

    # logger.debug("df.shape: " + str(raw_data.shape))

    # x_train, x_test, y_train = (
    #     train_data.drop(["Category"], axis=1),
    #     test_data.drop(["Category"], axis=1),
    #     train_data["Category"],
    # )
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(["Category"], axis=1),
        data["Category"],
        random_state=0,
        test_size=0.2,
    )

    model = LinearSVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # 保存模型
    joblib.dump(model, f"./models/{country}.pkl")
    # result = pd.DataFrame({"Keyword": raw_data["Keyword"], "Category": y_pred})
    # 统计分类成功率
    result = {
        "Country": country,
        "Accuracy": str(int(accuracy_score(y_test, y_pred) * 100)) + "%",
        "Precision": str(int(precision_score(y_test, y_pred, average="weighted") * 100))
        + "%",
        "Recall": str(int(recall_score(y_test, y_pred, average="weighted") * 100))
        + "%",
        "F1": str(int(f1_score(y_test, y_pred, average="weighted") * 100)) + "%",
    }

    return result
