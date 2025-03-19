import time
from vocab import Vocab
from vector import Vector
from db import MilvusDB
import pandas as pd
from config import DATA_CONFIG, MILVUS_CONFIG
from util import setup_readline, get_input, timer_decorator


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

            results = self.search(context)
            for item in results:
                new_df = self.df[self.df['index'] == item['id']]
                if new_df.empty:
                    continue
                row = new_df.iloc[0]
                content = row.content
                print()
                print(f'distance: {item["distance"]:.4f}')
                print(f'content: {content}')
                print()
            print('-' * 100)


if __name__ == "__main__":
    search = SearchResult()
    search.user_input()
