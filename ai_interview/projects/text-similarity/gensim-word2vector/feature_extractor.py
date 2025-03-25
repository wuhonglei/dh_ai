from vocab import Vocab
from dataset import NewsDatasetCsv, NewsItem
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
from config import MILVUS_CONFIG, DATASET_CONFIG, VOCAB_CONFIG, CACHE_CONFIG
from utils.common import init_dir, get_device
from vectory import Vector
import time
from db import MilvusDB
from type_definitions import DataItem
from model import CBOWModel


def collate_fn(batch: List[NewsItem], vector: Vector) -> List[DataItem]:
    data_items: List[DataItem] = []
    for item in batch:
        sentence = item['content']
        content_embedding = vector.get_embedding(sentence)
        data_items.append(
            DataItem(index=item['index'], content_embedding=content_embedding))
    return data_items


def main():
    init_dir()
    vocab = Vocab()

    model = CBOWModel()
    model.load(CACHE_CONFIG.val_cbow_model_cache_path)
    model_config = model.get_config()
    vector = Vector(vocab, model)
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False,
                            num_workers=0, collate_fn=lambda batch: collate_fn(batch, vector))

    total = 0
    start_time = time.time()
    tqdm_dataloader = tqdm(dataloader)
    db = MilvusDB(dimension=model_config['embedding_dim'],
                  milvus_config=MILVUS_CONFIG)
    for batch in tqdm_dataloader:
        db.insert(batch)
        total += len(batch)
    end_time = time.time()
    print(f"处理完成，共插入 {total} 条记录，耗时{end_time-start_time:.2f}秒")


if __name__ == "__main__":
    main()
