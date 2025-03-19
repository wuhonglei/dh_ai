from playwright.sync_api import sync_playwright, Page, Locator  # type: ignore
import logging
from typing import TypedDict, List
from tqdm import tqdm
import time
from utils import set_logger, read_js_file, write_nd_json, read_nd_json, hit_cache, write_json, read_json
from type_definitions import CategoryItem, NewsItem, create_category_item

set_logger()

logger = logging.getLogger(__name__)


category_list: list[CategoryItem] = []

cache_path = './cache/news_details.ndjson'


def goto_page(page: Page, url: str):
    try:
        page.goto(url, timeout=10000)
        time.sleep(2)
    except Exception as e:
        logger.error(f"访问 {url} 失败: {e}")


def main():
    with sync_playwright() as p:
        # 1. 打开浏览器
        browser = p.chromium.launch(headless=False)
        # 2. 打开页面
        page = browser.new_page()

        cache_category_list = read_json('result.json')
        if cache_category_list:
            result_list = cache_category_list
        else:
            # 3. 打开首页
            url = "https://news.sohu.com/"
            logger.info(f"打开首页: {url}")
            goto_page(page, url)

            # 4. 获取类目
            category_js_code = read_js_file("./lib/category.js")
            result_list = page.evaluate(category_js_code)
            logger.info(f"获取类目: {result_list}")

        category_list = [create_category_item(**item) for item in result_list]

        # 5. 获取新闻列表
        for category_item in tqdm(category_list, desc="获取新闻类目"):
            if category_item['news_list']:
                logger.info(f"新闻列表已存在: {category_item['category']}")
                continue

            logger.info(
                f"访问 {category_item['category']} 获取新闻列表: {category_item['url']}")
            goto_page(page, category_item['url'])
            news_list_js_code = read_js_file("./lib/news_list.js")
            logger.info(f"执行js代码")
            news_url_list: list[str] = page.evaluate(news_list_js_code)
            logger.info(f"获取新闻列表: {news_url_list}")
            if not news_url_list:
                logger.info(f"获取新闻列表失败: {category_item['category']}")
                continue

            json_list = read_nd_json(cache_path)
            for news_url in tqdm(news_url_list, desc="获取新闻详情"):
                if 'news_list' not in category_item:
                    category_item['news_list'] = []

                if hit_cache(news_url, json_list):
                    logger.info(f"新闻详情已存在: {news_url}")
                    category_item['news_list'].append(news_item)
                    continue

                logger.info(f"访问新闻详情: {news_url}")
                goto_page(page, news_url)

                news_item_js_code = read_js_file("./lib/news_item.js")
                logger.info(f"执行js代码")
                news_item: NewsItem = page.evaluate(news_item_js_code)
                logger.info(f"获取新闻详情: {news_item}")
                category_item['news_list'].append(news_item)
                write_nd_json(cache_path, news_item)  # type: ignore

        browser.close()
        write_json('result.json', category_list)


if __name__ == "__main__":
    main()
