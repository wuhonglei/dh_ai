import copy
import time
from search import SearchResult
from utils.common import load_json_file, write_json_file
from type_definitions import EvaluateResult, CategoryItem, EvaluateResultItem, create_evaluate_result_item
from tqdm import tqdm

from config import EVALUATE_CONFIG, EvaluateConfig, config


class EvaluateNews:
    def __init__(self, evaluate_config: EvaluateConfig | None = None):
        self.evaluate_config = evaluate_config or EVALUATE_CONFIG
        self.search = SearchResult()
        self.test_data = self.load_test_data()
        self.evaluate_result = self.load_evaluate_result()

    def load_test_data(self) -> list[CategoryItem]:
        return load_json_file(self.evaluate_config.test_data_path) or []

    def load_evaluate_result(self) -> EvaluateResult:
        cache = load_json_file(self.evaluate_config.evaluate_result_path)
        if cache is None:
            return {
                "meta": config.model_dump(),
                "results": [],
            }
        return cache

    def hit_cache(self, category_item: CategoryItem) -> bool:
        cached_category_names = [
            item['category']
            for item in self.evaluate_result['results']
            if item['news_list'] and all(news_item['search_result'] for news_item in item['news_list'])
        ]
        return category_item['category'] in cached_category_names

    def evaluate(self, save_cache: bool = False) -> EvaluateResult:
        start_time = time.time()
        for _category_item in tqdm(self.test_data, desc="evaluate category"):
            if self.hit_cache(_category_item):
                print(f"hit cache: {_category_item['category']}")
                continue

            category_item = create_evaluate_result_item(
                **_category_item)

            content_list: list[str] = [
                news_item['content']
                for news_item in category_item['news_list']
            ]
            if not content_list:
                print(f"no content: {category_item['category']}")
                continue

            print('search category: ', category_item['category'])
            search_results = self.search.search_with_content(content_list)
            for news_item, search_result in zip(category_item['news_list'], search_results):
                news_item['search_result'].extend(search_result)
            self.evaluate_result['results'].append(category_item)

            if save_cache:
                self.save_evaluate_result()
        end_time = time.time()
        print(f"evaluate time: {end_time - start_time:.2f} seconds")
        return self.evaluate_result

    def save_evaluate_result(self):
        write_json_file(self.evaluate_config.evaluate_result_path,
                        self.evaluate_result)


if __name__ == "__main__":
    evaluate = EvaluateNews()
    evaluate.evaluate(save_cache=True)
