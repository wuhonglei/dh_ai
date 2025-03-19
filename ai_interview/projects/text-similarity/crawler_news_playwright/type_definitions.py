
from typing import TypedDict


class NewsItem(TypedDict):
    title: str
    content: str
    url: str


class CategoryItem(TypedDict):
    category: str
    url: str
    news_list: list[NewsItem]


def create_category_item(category: str, url: str, news_list: list[NewsItem] | None = None) -> CategoryItem:
    return {
        "category": category,
        "url": url,
        "news_list": news_list or []
    }
