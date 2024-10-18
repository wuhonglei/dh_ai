import logging
import pathlib

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.utils import timer

logger = logging.getLogger(__name__)


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "imp_level1_category_1d",
        "pv_level1_category_1d",
        "order_level1_category_1d",
    ]
    df[cols] = df[cols].replace(r"^\s*$", "None", regex=True)
    df[cols] = df[cols].fillna("None")

    return df


def clean_text(df: pd.DataFrame, stopwords: str) -> pd.DataFrame:
    df["Keyword_Filter"] = df["Keyword_Filter"].astype(str)

    # Lower casing
    df["Keyword_Filter"] = df["Keyword_Filter"].str.lower()

    # Punctuation removal
    df["Keyword_Filter"] = (
        df["Keyword_Filter"]
        .str.replace("]", "", regex=False)
        .replace("[", "", regex=False)
        .replace("/", "", regex=False)
    )
    df["Keyword_Filter"] = df["Keyword_Filter"].str.replace("[^\w\s]", "", regex=True)

    # Remove all single characters
    df["Keyword_Filter"] = df["Keyword_Filter"].str.replace(
        "\s+[a-zA-Z]\s+|\^[a-zA-Z]\s+|\s+[a-zA-Z]$", "", regex=True
    )

    # Remove stopwords
    path = pathlib.Path("config") / "stopwords_custom.txt"
    with open(path, "r") as st:
        st_list = set(st.read().split())
    stop_words = nltk.corpus.stopwords.words(stopwords) + list(st_list)
    df["Keyword_Filter"] = df["Keyword_Filter"].apply(
        lambda x: [item for item in nltk.word_tokenize(x) if item not in stop_words]
    )
    df["Keyword_Filter"] = df["Keyword_Filter"].apply(lambda x: " ".join(x))

    return df


def vectorize(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = CountVectorizer()
    raw_text = vectorizer.fit_transform(df["Keyword_Filter"])
    df_vectorize = pd.DataFrame(
        raw_text.toarray(), columns=vectorizer.get_feature_names_out()
    )
    return df_vectorize


@timer
def pre_process(data: pd.DataFrame, stopwords: str) -> pd.DataFrame:
    data = fill_missing_data(data)
    data_preprocess = clean_text(data, stopwords)
    data_vectorizer = vectorize(data_preprocess)

    data_onehot = pd.get_dummies(
        data[
            [
                "Category",
                "imp_level1_category_1d",
                "pv_level1_category_1d",
                "order_level1_category_1d",
            ]
        ],
        columns=[
            "imp_level1_category_1d",
            "pv_level1_category_1d",
            "order_level1_category_1d",
        ],
        dummy_na=True,
    )
    data_onehot.reset_index(inplace=True, drop=True)

    df_end = pd.concat([data_onehot, data_vectorizer], axis=1)

    return df_end
