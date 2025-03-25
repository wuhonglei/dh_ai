import pandas as pd
from search import SearchResult
from utils.common import get_device
from type_definitions import DbResult
from tqdm import tqdm
from vocab import Vocab
from db import MilvusDB
from vectory import Vector
from config import EVALUATE_TITLE_CONFIG, EvaluateTitleConfig, CACHE_CONFIG, VOCAB_CONFIG, DATASET_CONFIG, MILVUS_CONFIG
from model import SiameseNetwork
import torch


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

    def print_performance(self):
        df = pd.read_csv(self.evaluate_config.evaluate_result_path)
        df['rank'] = 1 / df['rank']
        print(f'MRR', df['rank'].mean())


if __name__ == "__main__":
    bert_name = VOCAB_CONFIG.bert_name
    max_length = VOCAB_CONFIG.max_length
    embedding_dim = VOCAB_CONFIG.embedding_dim
    val_csv_path = DATASET_CONFIG.val_csv_path
    projection_dim = VOCAB_CONFIG.projection_dim
    val_cbow_model_cache_path = CACHE_CONFIG.val_cbow_model_cache_path

    vocab = Vocab(bert_name, max_length)
    device = get_device()
    db = MilvusDB(dimension=embedding_dim, milvus_config=MILVUS_CONFIG)
    df = pd.read_csv(val_csv_path)
    model = SiameseNetwork(bert_name, max_length, projection_dim)
    model.load_state_dict(torch.load(
        val_cbow_model_cache_path, map_location=device))
    model.to(device)
    model.eval()
    vector = Vector(type="title", vocab=vocab, model=model, device=device)
    evaluate = EvaluateTitle(
        vector, db, df, EVALUATE_TITLE_CONFIG, MILVUS_CONFIG.version)
    evaluate.evaluate()
    evaluate.save_evaluate_result()
    evaluate.print_performance()
