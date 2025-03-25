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

    def __iter__(self) -> Iterator[list[str]]:
        for sentence_data in self.dataset:
            words = self.process_sentence(sentence_data)
            yield words


def generate_cbow_dataset_cache():
    min_freq = VOCAB_CONFIG.min_freq
    max_freq = VOCAB_CONFIG.max_freq
    window = VOCAB_CONFIG.window
    cache_path = CACHE_CONFIG.val_cbow_dataset_cache_path

    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    cache_path = get_train_dataset_cache_path(min_freq, max_freq, window)
    cbow_dataset = CBOWDataset(dataset, vocab, window, cache_path)
    print(len(cbow_dataset))
    print(cbow_dataset[0])


def check_cbow_dataset_cache():
    files = []
    vocab = Vocab(VOCAB_CONFIG)
    for file in os.listdir('./cache/'):
        if file.endswith('.pkl') and file.startswith('train_dataset_'):
            files.append(file)
    error_files = [
        './cache/train_dataset_cache_350_105190_8.pkl'
    ]
    for file in tqdm(files, desc="检查 CBOW 数据集缓存"):
        cache_path = os.path.join('./cache/', file)
        if cache_path not in error_files:
            continue

        filename = cache_path.split('.')[-2]
        window = int(filename.split('_')[-1])
        cbow_dataset = CBOWDataset(
            None, vocab, window, cache_path)  # type: ignore
        for i in tqdm(range(len(cbow_dataset)), desc=f"检查 CBOW 数据集缓存 {filename}"):
            context, target = cbow_dataset[i]
            if context.size(0) != 2 * window:
                print(f"CBOW 数据集缓存 {file} 第 {i} 个样本的上下文窗口大小不一致")
            if target is None or not target:
                print(f"CBOW 数据集缓存 {file} 第 {i} 个样本的 target 为 None")


if __name__ == "__main__":
    # generate_cbow_dataset_cache()
    check_cbow_dataset_cache()
