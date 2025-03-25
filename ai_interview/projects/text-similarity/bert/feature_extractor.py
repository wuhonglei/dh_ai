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
from model import SiameseNetwork
import torch


def collate_fn(batch: List[NewsItem], vector: Vector) -> List[DataItem]:
    data_items: List[DataItem] = []
    content_list = [item['content'] for item in batch]
    content_embeddings = vector.get_embeddings(content_list)
    for item, content_embedding in zip(batch, content_embeddings):
        data_items.append(
            DataItem(index=item['index'], content_embedding=content_embedding))
    return data_items


def main():
    init_dir()
    bert_name = VOCAB_CONFIG.bert_name
    embedding_dim = VOCAB_CONFIG.embedding_dim
    max_length = VOCAB_CONFIG.max_length

    vocab = Vocab(bert_name, max_length=max_length)
    device = get_device()
    model = SiameseNetwork(bert_name, max_position_embeddings=max_length,
                           use_projection=VOCAB_CONFIG.use_projection)
    print(f'load model from {CACHE_CONFIG.val_cbow_model_cache_path}')
    model.load_state_dict(torch.load(
        CACHE_CONFIG.val_cbow_model_cache_path, map_location=device))
    model.to(device)
    model.eval()
    vector = Vector(vocab, model, device)
    dataset = NewsDatasetCsv(DATASET_CONFIG.train_csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=lambda batch: collate_fn(batch, vector))

    total = 0
    start_time = time.time()
    tqdm_dataloader = tqdm(dataloader)
    db = MilvusDB(dimension=embedding_dim, milvus_config=MILVUS_CONFIG)
    for batch in tqdm_dataloader:
        db.insert(batch)
        total += len(batch)
    end_time = time.time()
    print(f"处理完成，共插入 {total} 条记录，耗时{end_time-start_time:.2f}秒")


if __name__ == "__main__":
    main()
