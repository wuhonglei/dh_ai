from vocab import Vocab
from vector import Vector
from db import MilvusDB
import pandas as pd
from config import DATA_CONFIG, MILVUS_CONFIG
from util import setup_readline, get_input


def search():
    vocab = Vocab()
    vocab.load_vocab_from_txt(DATA_CONFIG.vocab_path,
                              min_freq=DATA_CONFIG.min_freq)
    vector = Vector(vocab)

    db = MilvusDB(dimension=len(vocab), milvus_config=MILVUS_CONFIG)

    df = pd.read_csv(DATA_CONFIG.val_csv_path)

    # 在主循环之前调用
    setup_readline()

    while True:
        context = get_input()
        if context == 'q':
            break
        if context is None or len(context) == 0:
            continue

        print('开始搜索...')
        embedding = vector.vectorize_text(context)
        results = db.search(embedding, limit=3)
        for item in results:
            new_df = df[df['index'] == item['id']]
            if new_df.empty:
                continue
            row = new_df.iloc[0]
            content = row.content
            print()
            print(f'distance: {item["distance"]:.4f}')
            print(f'content: {content}')
            print()
        print('-' * 100)

        # 替换原来的输入
        context = get_input()


if __name__ == "__main__":
    search()
