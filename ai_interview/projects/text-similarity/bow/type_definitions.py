
from typing import TypedDict, Optional
from numpy.typing import NDArray
import numpy as np
from config import AppConfig


class NewsItem(TypedDict):
    title: str
    content: str
    url: str


class CategoryItem(TypedDict):
    category: str
    url: str
    news_list: list[NewsItem]


class DataItem(TypedDict):
    index: int
    content_embedding: NDArray[np.float16]


class DbResult(TypedDict):
    id: int
    distance: float


class DbResultWithContent(DbResult):
    content: str


class SearchResultItem(DbResult):
    content: str


class EvaluateNewsItem(NewsItem):
    search_result: list[SearchResultItem]


class EvaluateResultItem(CategoryItem):
    news_list: list[EvaluateNewsItem]


class EvaluateResult(TypedDict):
    meta: AppConfig
    results: list[EvaluateResultItem]


class CsvRow(TypedDict):
    index: int
    title: str
    content: Optional[str]


def create_category_item(category: str, url: str, news_list: list[NewsItem] | None = None) -> CategoryItem:
    return {
        "category": category,
        "url": url,
        "news_list": news_list or []
    }


def create_evaluate_news_item(news: NewsItem) -> EvaluateNewsItem:
    return {
        "url": news["url"],
        "title": news["title"],
        "content": news["content"],
        "search_result": []
    }


def create_evaluate_result_item(category: str, url: str, news_list: list[NewsItem]) -> EvaluateResultItem:
    _news_list = news_list or []
    new_news_list: list[EvaluateNewsItem] = [
        create_evaluate_news_item(news) for news in _news_list]

    return {
        "category": category,
        "url": url,
        "news_list": new_news_list
    }
