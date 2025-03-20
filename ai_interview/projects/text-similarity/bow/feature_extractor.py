from vocab import Vocab
from vector import Vector
from dataset import NewsDatasetCsv, NewsItem
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
from config import MILVUS_CONFIG, DATASET_CONFIG, VOCAB_CONFIG
from utils.common import init_dir
import time
from db import MilvusDB

from type_definitions import DataItem


def collate_fn(batch: List[NewsItem], vector: Vector) -> List[DataItem]:
    data_items: List[DataItem] = []
    for item in batch:
        content = item['content']
        content_embedding = vector.vectorize_text(content)
        data_items.append(
            DataItem(index=item['index'], content_embedding=content_embedding))
    return data_items


def main():
    init_dir()
    vocab = Vocab()
    vocab.load_vocab_from_txt()
    print('embedding size:', len(vocab.word_to_index))
    vector = Vector(vocab)
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False,
                            num_workers=0, collate_fn=lambda batch: collate_fn(batch, vector))

    total = 0
    start_time = time.time()
    tqdm_dataloader = tqdm(dataloader)
    db = MilvusDB(dimension=len(vocab), milvus_config=MILVUS_CONFIG)
    for batch in tqdm_dataloader:
        db.insert(batch)
        total += len(batch)
    end_time = time.time()
    print(f"处理完成，共插入 {total} 条记录，耗时{end_time-start_time:.2f}秒")


if __name__ == "__main__":
    main()
