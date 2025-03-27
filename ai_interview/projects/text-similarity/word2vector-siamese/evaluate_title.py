import pandas as pd
from search import SearchResult
from utils.common import get_device
from type_definitions import DbResult
from tqdm import tqdm
from vocab import Vocab
from db import MilvusDB
from vectory import Vector
from config import EVALUATE_TITLE_CONFIG, EvaluateTitleConfig, CACHE_CONFIG, VOCAB_CONFIG, DATASET_CONFIG, VERSION, MILVUS_CONFIG
from model import SiameseNetwork
import torch


class EvaluateTitle:
    def __init__(self, vector: Vector, db: MilvusDB, df: pd.DataFrame, evaluate_config: EvaluateTitleConfig, version: str, max_title_length: int):
        self.evaluate_config = evaluate_config
        self.version = version
        self.max_title_length = max_title_length

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
                batch_title, max_length=self.max_title_length, limit=self.evaluate_config.limit)

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
    vocab = Vocab()
    vocab.load_vocab_from_txt()
    device = get_device()
    pad_idx = vocab.pad_idx
    vocab_size = len(vocab)
    embedding_dim = VOCAB_CONFIG.embedding_dim
    hidden_dim = VOCAB_CONFIG.hidden_dim
    projection_dim = VOCAB_CONFIG.projection_dim
    best_model_path = CACHE_CONFIG.best_model_path
    test_csv_path = DATASET_CONFIG.test_csv_path
    max_title_length = VOCAB_CONFIG.max_title_length

    db = MilvusDB(dimension=projection_dim, milvus_config=MILVUS_CONFIG)
    df = pd.read_csv(test_csv_path)

    model = SiameseNetwork(vocab_size, embedding_dim,
                           hidden_dim, projection_dim, pad_idx)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    vector = Vector(vocab, model, device)
    evaluate = EvaluateTitle(
        vector, db, df, EVALUATE_TITLE_CONFIG, VERSION, max_title_length)
    evaluate.evaluate()
    evaluate.save_evaluate_result()
    evaluate.print_performance()
