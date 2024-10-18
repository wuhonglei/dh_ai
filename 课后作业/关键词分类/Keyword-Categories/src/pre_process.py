import logging
import pathlib

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.utils import timer

logger = logging.getLogger(__name__)


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Category",
    ]
    df[cols] = df[cols].replace(
        r"^\s*$", "None", regex=True)  # 将空白字符或连续空格替换为None
    df[cols] = df[cols].fillna("None")

    return df


def clean_text(df: pd.DataFrame, stopwords: str) -> pd.DataFrame:
    df["Keyword"] = df["Keyword"].astype(str)

    # Lower casing
    df["Keyword"] = df["Keyword"].str.lower()

    # Punctuation removal
    df["Keyword"] = (
        df["Keyword"]
        .str.replace("]", "", regex=False)
        .replace("[", "", regex=False)
        .replace("/", "", regex=False)
    )
    df["Keyword"] = df["Keyword"].str.replace(
        "[^\w\s]", "", regex=True)

    # Remove all single characters
    df["Keyword"] = df["Keyword"].str.replace(
        "\s+[a-zA-Z]\s+|\^[a-zA-Z]\s+|\s+[a-zA-Z]$", "", regex=True
    )

    # Remove stopwords
    path = pathlib.Path("config") / "stopwords_custom.txt"
    with open(path, "r") as st:
        st_list = set(st.read().split())
    stop_words = nltk.corpus.stopwords.words(stopwords) + list(st_list)
    df["Keyword"] = df["Keyword"].apply(
        lambda x: [item for item in nltk.word_tokenize(
            x) if item not in stop_words]
    )
    df["Keyword"] = df["Keyword"].apply(lambda x: " ".join(x))

    return df


def vectorize(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = CountVectorizer()
    raw_text = vectorizer.fit_transform(df["Keyword"])
    df_vectorize = pd.DataFrame(
        raw_text.toarray(), columns=vectorizer.get_feature_names_out()
    )
    return df_vectorize


@timer
def pre_process(data: pd.DataFrame, stopwords: str) -> pd.DataFrame:
    data = fill_missing_data(data)
    data_preprocess = clean_text(data, stopwords)
    data_vectorizer = vectorize(data_preprocess)

    # data_onehot = pd.get_dummies(
    #     data[
    #         [
    #             "Category",
    #         ]
    #     ],
    #     columns=[
    #         "Category",
    #     ],
    #     dummy_na=True,
    # )
    # data_onehot.reset_index(inplace=True, drop=True)

    df_end = pd.concat([data['Category'], data_vectorizer], axis=1)

    return df_end
