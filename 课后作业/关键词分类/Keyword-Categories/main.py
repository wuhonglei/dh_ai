from typing import List
import logging
import pathlib

import nltk

from config.config import settings
from src.pre_process import pre_process
from src.train_model import model_selection, train_model
from src.utils import get_gsheet_df, update_gsheet_df, timer, get_df_from_xlsx

logger = logging.getLogger(__name__)
nltk.download("stopwords")


class KeywordCategories:
    def __init__(self, info: dict[str, str]) -> None:
        self.country = info["country"]
        self.stopwords = info["stopwords"]

    @timer
    def main(self) -> None:

        print(f"{'-'*6}{self.country}{'-'*6}")

        # Step1 > Get gsheet raw data
        # raw_data = get_gsheet_df("Result-" + self.country)
        raw_data = get_df_from_xlsx(
            './data/Keyword Categorization.xlsx', self.country)

        # Step2 > Pre-process data
        process_data = pre_process(raw_data, self.stopwords)

        # (Option) > Model selection
        # model_selection(self.country, process_data)

        # Step3 > Train model & predict
        result = train_model(self.country, process_data, raw_data)
        result.to_csv(f'./result/{self.country}.csv')

        # Step4 > Save to Google Sheet
        # update_gsheet_df("Predict-Result-" + self.country, result)


if __name__ == "__main__":

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

    path = pathlib.Path().resolve() / "csv" / "predict_result.csv"
    if path.exists():
        path.unlink()

    for info in countries_info:
        categories = KeywordCategories(info)
        categories.main()
