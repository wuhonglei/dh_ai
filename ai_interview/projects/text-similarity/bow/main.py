from vocab import Vocab
from vector import Vector
from db import MilvusDB, DataItem
import pandas as pd
from dataset import NewsDatasetCsv
from torch.utils.data import DataLoader


def collate_fn(batch):
    return batch


def main():
    vocab = Vocab()
    vocab.load_vocab_from_txt("../data/vocab.txt", min_freq=90)
    vector = Vector(vocab)

    db = MilvusDB(db_name="./database/milvus_demo.db",
                  collection_name="bow_title_content_collection", dimension=len(vocab))

    dataset = NewsDatasetCsv("../data/val.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)


if __name__ == "__main__":
    main()
