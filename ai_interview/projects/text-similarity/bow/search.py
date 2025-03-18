from torch import embedding
from vocab import Vocab
from vector import Vector
from db import MilvusDB, DataItem
import pandas as pd
from dataset import NewsDatasetCsv, NewsItem
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
from config import BowConfig
import readline


def setup_readline():
    # 设置历史文件路径
    readline.parse_and_bind('tab: complete')
    # 设置历史记录长度
    readline.set_history_length(1000)
    try:
        # 尝试从历史文件加载
        readline.read_history_file(BowConfig.search_history_path)
    except FileNotFoundError:
        pass


def get_input() -> str | None:
    try:
        context = input('请输入搜索内容: ')
        # 保存到历史文件
        readline.write_history_file(BowConfig.search_history_path)
        return context
    except KeyboardInterrupt:
        print('\n已取消输入')
        return None


def search():
    vocab = Vocab()
    vocab.load_vocab_from_txt(BowConfig.vocab_path,
                              min_freq=BowConfig.min_freq)
    vector = Vector(vocab)

    db = MilvusDB(db_name=BowConfig.db_name,
                  collection_name=BowConfig.collection_name, dimension=len(vocab))

    # 在主循环之前调用
    setup_readline()

    while True:
        context = get_input()
        if context is None or context == 'q':
            break

        print('开始搜索...')
        embedding = vector.vectorize_text(context)
        results = db.search(embedding, limit=10)
        print(results)
        # 替换原来的输入
        context = get_input()


if __name__ == "__main__":
    search()
