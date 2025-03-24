import time
from vocab import Vocab
from model import SiameseNetwork
from db import MilvusDB
import pandas as pd
from config import DATASET_CONFIG, MILVUS_CONFIG, MilvusConfig, DataSetConfig, VocabConfig, VOCAB_CONFIG
from utils.common import setup_readline, get_input, timer_decorator, load_json_file
from type_definitions import CsvRow, DbResultWithContent, DbResult
from utils.common import get_device
from config import CACHE_CONFIG
from vectory import Vector


class SearchResult:
    def __init__(self, vector: Vector, db: MilvusDB, df: pd.DataFrame):
        self.vector = vector
        self.db = db
        self.df = df

    @timer_decorator
    def search(self, context: list[str], limit: int = 3) -> list[list[DbResult]]:
        embeddings = []
        for sentence in context:
            embedding = self.vector.get_embedding(sentence)
            embeddings.append(embedding)

        results = self.db.search(embeddings, limit=limit)
        return results

    def search_with_content(self, context: list[str], limit: int = 3) -> list[list[DbResultWithContent]]:
        search_results = self.search(context, limit)  # type: ignore
        ans: list[list[DbResultWithContent]] = []
        for search_result in search_results:
            temp: list[DbResultWithContent] = []
            ids = [item['id'] for item in search_result]
            rows = self.get_rows_by_id(ids)
            for search_result, row in zip(search_result, rows):
                temp.append(DbResultWithContent(
                    **search_result, content=row['content'] or ''))
            ans.append(temp)
        return ans

    @timer_decorator
    def get_rows_by_id(self, ids: list[int]) -> list[CsvRow]:
        new_df = self.df[self.df['index'].isin(ids)]
        return new_df.to_dict(orient='records')  # type: ignore

    def user_input(self, limit: int = 3):
        setup_readline()
        while True:
            context = get_input()
            if context == 'q':
                break
            if context is None or len(context) == 0:
                print("输入为空，请重新输入")
                continue

            results = self.search_with_content([context], limit)
            for item in results[0]:
                print()
                print(f'distance: {item["distance"]:.4f}')
                print(f'content: {item["content"]}')
                print()
            print('-' * 100)


if __name__ == "__main__":
    bert_name = VOCAB_CONFIG.bert_name
    max_length = VOCAB_CONFIG.max_length
    embedding_dim = VOCAB_CONFIG.embedding_dim
    val_csv_path = DATASET_CONFIG.val_csv_path
    use_projection = VOCAB_CONFIG.use_projection

    vocab = Vocab(bert_name, max_length)
    device = get_device()
    db = MilvusDB(dimension=embedding_dim, milvus_config=MILVUS_CONFIG)
    df = pd.read_csv(val_csv_path)
    model = SiameseNetwork(bert_name, max_length, use_projection)
    model.to(device)
    model.eval()
    vector = Vector(vocab, model, device)
    search = SearchResult(vector, db, df)
    search.user_input()
