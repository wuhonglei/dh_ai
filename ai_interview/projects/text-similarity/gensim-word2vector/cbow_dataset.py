import os
from torch.utils.data import Dataset
from dataset import NewsDatasetCsv
from vocab import Vocab
from utils.train import get_train_dataset_cache_path
from tqdm import tqdm
from config import DATASET_CONFIG, CACHE_CONFIG, VOCAB_CONFIG
from typing import Iterator


class CBOWDataset:
    """
    生成 CBOW 的训练数据
    """

    def __init__(self, dataset: NewsDatasetCsv, vocab: Vocab):
        self.dataset = dataset
        self.vocab = vocab

    def process_sentence(self, sentence_data) -> list[str]:
        """处理单个句子的辅助函数"""
        sentence = sentence_data['content']
        words = self.vocab.tokenize(sentence)
        return words

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[list[str]]:
        for sentence_data in self.dataset:
            words = self.process_sentence(sentence_data)
            yield words
