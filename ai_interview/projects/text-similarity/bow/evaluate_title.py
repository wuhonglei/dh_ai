import copy
import enum
import time
import pandas as pd
from search import SearchResult
from utils.common import load_json_file, write_json_file
from type_definitions import EvaluateResult, CategoryItem, EvaluateResultItem, create_evaluate_result_item, DbResult
from tqdm import tqdm

from config import EVALUATE_TITLE_CONFIG, EvaluateTitleConfig, MILVUS_CONFIG, config, MilvusConfig


class EvaluateTitle:
    def __init__(self, evaluate_config: EvaluateTitleConfig | None = None, milvus_config: MilvusConfig | None = None):
        self.evaluate_config = evaluate_config or EVALUATE_TITLE_CONFIG
        self.milvus_config = milvus_config or MILVUS_CONFIG
        self.version = self.milvus_config.version

        print('milvus_version', self.version)
        print('evaluate_result_path', self.evaluate_config.evaluate_result_path)

        self.search = SearchResult()
        self.test_data = self.load_test_data()
        self.evaluate_result = self.load_evaluate_result()

    def load_test_data(self):
        return pd.read_csv(self.evaluate_config.test_data_path)

    def load_evaluate_result(self):
        head = ['index', 'title', 'version_name', 'rank', 'score', 'content']
        return pd.DataFrame(columns=head)

    def evaluate(self):
        total_count = len(self.test_data)
        batch_size = 100
        result_list = []
        progress_bar = tqdm(total=total_count, desc="Evaluating")

        for i in range(0, total_count, batch_size):
            batch_data = self.test_data.iloc[i:i+batch_size]
            progress_bar.update(len(batch_data))
            batch_title = batch_data['title'].tolist()
            search_results = self.search.search(
                batch_title, limit=self.evaluate_config.limit)

            # 遍历每个样本
            for idx, (index, row) in enumerate(batch_data.iterrows()):
                search_result: list[DbResult] = search_results[idx]

                rank = 10000  # rank = 1 表示最高的排名
                score = 0
                # 遍历每个样本的 tok_k 个结果
                for rank_index, result in enumerate(search_result):
                    if result['id'] == row['index']:
                        rank = rank_index + 1
                        score = result['distance']
                        break

                result_item = {
                    'index': row['index'],
                    'title': row['title'],
                    'version_name': self.version,
                    'rank': rank,
                    'score': score,
                    'content': row['content']
                }
                result_list.append(result_item)

        self.evaluate_result = pd.DataFrame(result_list)

    def save_evaluate_result(self):
        self.evaluate_result.to_csv(
            self.evaluate_config.evaluate_result_path, index=False)


if __name__ == "__main__":
    evaluate = EvaluateTitle()
    evaluate.evaluate()
    evaluate.save_evaluate_result()
