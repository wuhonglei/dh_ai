import copy
import enum
import time
import pandas as pd
from search import SearchResult
from utils.common import load_json_file, write_json_file, get_device, load_model
from type_definitions import EvaluateResult, CategoryItem, EvaluateResultItem, create_evaluate_result_item, DbResult
from tqdm import tqdm
from vocab import Vocab
from db import MilvusDB
from vectory import Vector
from config import EVALUATE_TITLE_CONFIG, EvaluateTitleConfig, CACHE_CONFIG, VOCAB_CONFIG, DATASET_CONFIG, MilvusConfig, MILVUS_CONFIG


class EvaluateTitle:
    def __init__(self, vector: Vector, db: MilvusDB, df: pd.DataFrame, evaluate_config: EvaluateTitleConfig, version: str):
        self.evaluate_config = evaluate_config
        self.version = version

        print('milvus_version', self.version)
        print('evaluate_result_path', self.evaluate_config.evaluate_result_path)

        self.search = SearchResult(vector, db, df)
        self.test_data = self.load_test_data()
        self.evaluate_result = self.load_evaluate_result()

    def load_test_data(self):
        return pd.read_csv(self.evaluate_config.test_data_path)

    def load_evaluate_result(self):
        head = ['index', 'title', 'version_name', 'rank', 'score', 'content']
        return pd.DataFrame(columns=head)

    def evaluate(self):
        batch_size = 100
        result_list = []
        total_count = len(self.test_data)
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
    vocab = Vocab()
    vocab.load_vocab_from_txt()
    device = get_device()
    embedding_dim = VOCAB_CONFIG.embedding_dim
    db = MilvusDB(dimension=embedding_dim, milvus_config=MILVUS_CONFIG)
    df = pd.read_csv(DATASET_CONFIG.val_csv_path)
    model = load_model(len(vocab), vocab.pad_idx, embedding_dim,
                       CACHE_CONFIG.val_cbow_model_cache_path)
    model.to(device)
    model.eval()
    vector = Vector(vocab, model, device)
    evaluate = EvaluateTitle(
        vector, db, df, EVALUATE_TITLE_CONFIG, MILVUS_CONFIG.version)
    evaluate.evaluate()
    evaluate.save_evaluate_result()
