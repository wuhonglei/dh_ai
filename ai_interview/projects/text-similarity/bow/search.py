import time
from vocab import Vocab
from vector import Vector
from db import MilvusDB
import pandas as pd
from config import DATASET_CONFIG, MILVUS_CONFIG, MilvusConfig, DataSetConfig, VocabConfig, VOCAB_CONFIG
from utils.common import setup_readline, get_input, timer_decorator, load_json_file
from type_definitions import CsvRow, DbResultWithContent, DbResult


class SearchResult:
    def __init__(self, milvus_config: MilvusConfig | None = None, dataset_config: DataSetConfig | None = None, vocab_config: VocabConfig | None = None):
        self.milvus_config = milvus_config or MILVUS_CONFIG
        self.dataset_config = dataset_config or DATASET_CONFIG
        self.vocab_config = vocab_config or VOCAB_CONFIG

        self.vocab = Vocab()
        self.vocab.load_vocab_from_txt()
        idf_dict = load_json_file(self.vocab_config.word_idf_path)
        self.vector = Vector(self.vocab, idf_dict)
        self.db = MilvusDB(dimension=len(self.vocab),
                           milvus_config=self.milvus_config)
        self.df = pd.read_csv(self.dataset_config.val_csv_path)

    @timer_decorator
    def search(self, context: list[str], limit: int = 3) -> list[list[DbResult]]:
        embeddings = self.vector.batch_vectorize_text(context)
        # 添加维度检查
        expected_dim = len(self.vocab)
        actual_dim = len(embeddings[0])
        if expected_dim != actual_dim:
            raise ValueError(
                f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")

        results = self.db.search(embeddings, limit=limit)
        return results

    def search_with_content(self, context: list[str]) -> list[list[DbResultWithContent]]:
        search_results = self.search(context)  # type: ignore
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

    def user_input(self):
        setup_readline()
        while True:
            context = get_input()
            if context == 'q':
                break
            if context is None or len(context) == 0:
                print("输入为空，请重新输入")
                continue

            results = self.search_with_content([context])
            for item in results[0]:
                print()
                print(f'distance: {item["distance"]:.4f}')
                print(f'content: {item["content"]}')
                print()
            print('-' * 100)


if __name__ == "__main__":
    search = SearchResult()
    search.user_input()
