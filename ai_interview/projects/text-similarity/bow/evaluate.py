import copy
from search import SearchResult
from util import load_json_file, write_json_file
from type_definitions import EvaluateResult, create_evaluate_result_item
from tqdm import tqdm

from config import EVALUATE_CONFIG, EvaluateConfig, config


class Evaluate:
    def __init__(self, evaluate_config: EvaluateConfig | None = None):
        self.evaluate_config = evaluate_config or EVALUATE_CONFIG
        self.search = SearchResult()
        self.test_data = self.load_test_data()
        self.evaluate_result: EvaluateResult = {
            "meta": config,
            "results": [],
        }

    def load_test_data(self):
        return load_json_file(self.evaluate_config.test_data_path)

    def evaluate(self) -> EvaluateResult:
        for category_item in tqdm(self.test_data, desc="evaluate category"):
            category_item = create_evaluate_result_item(
                **category_item)

            progress = tqdm(category_item['news_list'], desc="evaluate news")
            for news_item in progress:
                progress.write(f"evaluate news: {news_item['content']}")
                search_result = self.search.search_with_content(
                    news_item['content'])
                news_item['search_result'].extend(search_result)
                break
            self.evaluate_result['results'].append(category_item)
            break
        return self.evaluate_result

    def save_evaluate_result(self):
        write_json_file(self.evaluate_config.evaluate_result_path,
                        self.evaluate_result)


if __name__ == "__main__":
    evaluate = Evaluate()
    evaluate.evaluate()
    evaluate.save_evaluate_result()
