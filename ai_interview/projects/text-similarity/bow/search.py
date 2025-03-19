import time
from vocab import Vocab
from vector import Vector
from db import MilvusDB
import pandas as pd
from config import DATA_CONFIG, MILVUS_CONFIG
from util import setup_readline, get_input, timer_decorator
from type_definitions import CsvRow, DbResultWithContent


class SearchResult:
    def __init__(self):
        self.vocab = Vocab()
        self.vocab.load_vocab_from_txt(DATA_CONFIG.vocab_path,
                                       min_freq=DATA_CONFIG.min_freq)
        self.vector = Vector(self.vocab)
        self.db = MilvusDB(dimension=len(self.vocab),
                           milvus_config=MILVUS_CONFIG)
        self.df = pd.read_csv(DATA_CONFIG.val_csv_path)

    @timer_decorator
    def search(self, context: str):
        embedding = self.vector.vectorize_text(context)
        results = self.db.search([embedding], limit=3)
        return results[0]

    def search_with_content(self, context: str) -> list[DbResultWithContent]:
        search_results = self.search(context)  # type: ignore
        ids = [item['id'] for item in search_results]
        rows = self.get_rows_by_id(ids)
        ans: list[DbResultWithContent] = []
        for search_result, row in zip(search_results, rows):
            ans.append(DbResultWithContent(
                **search_result, content=row['content'] or ''))
        return ans

    @timer_decorator
    def get_rows_by_id(self, ids: list[int]) -> list[CsvRow]:
        new_df = self.df[self.df['index'].isin(ids)]
        return new_df.to_dict(orient='records')  # type: ignore

    @timer_decorator
    def batch_search(self, context: list[str]):
        embeddings = self.vector.batch_vectorize_text(context)
        results = self.db.search(embeddings, limit=3)
        return results

    def user_input(self):
        setup_readline()
        while True:
            context = get_input()
            if context == 'q':
                break
            if context is None or len(context) == 0:
                print("输入为空，请重新输入")
                continue

            results = self.search_with_content(context)
            for item in results:
                print()
                print(f'distance: {item["distance"]:.4f}')
                print(f'content: {item["content"]}')
                print()
            print('-' * 100)


if __name__ == "__main__":
    search = SearchResult()
    search.user_input()
