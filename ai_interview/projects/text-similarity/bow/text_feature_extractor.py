import os
from vocab import Vocab
from vector import Vector
from db import MilvusDB, DataItem
import pandas as pd
from dataset import NewsDatasetCsv, NewsItem
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
from config import MILVUS_CONFIG, DATA_CONFIG, CacheConfig


def collate_fn(batch: List[NewsItem], vector: Vector) -> List[DataItem]:
    data_items: List[DataItem] = []
    for item in batch:
        content = item['content']
        content_embedding = vector.vectorize_text(content)
        data_items.append(
            DataItem(index=item['index'], content_embedding=content_embedding))
    return data_items


def init():
    dir_path = [
        MILVUS_CONFIG.db_name,
        DATA_CONFIG.vocab_path,
        CacheConfig.search_history_path,
    ]
    for path in dir_path:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)


def main():
    init()
    vocab = Vocab()
    vocab.load_vocab_from_txt(DATA_CONFIG.vocab_path,
                              min_freq=DATA_CONFIG.min_freq)
    vector = Vector(vocab)

    db = MilvusDB(dimension=len(vocab), milvus_config=MILVUS_CONFIG)

    dataset = NewsDatasetCsv(DATA_CONFIG.val_csv_path)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False,
                            num_workers=0, collate_fn=lambda batch: collate_fn(batch, vector))

    tqdm_dataloader = tqdm(dataloader)
    total = 0
    for batch in tqdm_dataloader:
        db.insert(batch)
        total += len(batch)
        if total > 10:
            break
    print(f"Inserted {total} items into Milvus")


if __name__ == "__main__":
    main()
