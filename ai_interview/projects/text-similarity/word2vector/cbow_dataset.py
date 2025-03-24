import os
from torch.utils.data import Dataset
from dataset import NewsDatasetCsv
from vocab import Vocab
from utils.common import load_pickle_file, save_pickle_file, timer_decorator
from utils.train import get_train_dataset_cache_path
from tqdm import tqdm
from config import DATASET_CONFIG, CACHE_CONFIG, VOCAB_CONFIG
import torch
from multiprocessing import Pool, cpu_count


class CBOWDataset(Dataset):
    """
    生成 CBOW 的训练数据
    """

    def __init__(self, dataset: NewsDatasetCsv, vocab: Vocab, window_size: int, cache_path: str = ''):
        self.dataset = dataset
        self.window_size = window_size
        self.vocab = vocab
        self.cache_path = cache_path
        self.data = self.load_data()

    def process_sentence(self, sentence_data) -> list[tuple[list[int], int]]:
        """处理单个句子的辅助函数"""
        sentence = sentence_data['content']
        words = self.vocab.tokenize(sentence)
        indices = self.vocab.batch_encoder(words)
        if len(indices) <= self.window_size:
            return []

        result = []
        for i, target_index in enumerate(indices):
            context = indices[max(0, i - self.window_size):i] + \
                indices[i + 1:i + self.window_size + 1]
            if context:
                result.append((context, target_index))
        return result

    @timer_decorator
    def load_data(self):
        if self.cache_path:
            sentences = load_pickle_file(self.cache_path)
            if sentences:
                return sentences

        if self.dataset is None or self.vocab is None:
            raise ValueError("dataset 和 vocab 不能为 None")

        # 准备数据
        sentences = []
        for i in tqdm(range(len(self.dataset)), desc="生成 CBOW 训练数据"):
            sentences.extend(self.process_sentence(self.dataset[i]))

        if self.cache_path:
            save_pickle_file(self.cache_path, sentences)

        return sentences

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        context_idxs, target_idx = self.data[index]
        # 确保上下文窗口大小一致
        if len(context_idxs) < self.window_size * 2:
            context_idxs = [self.vocab.pad_idx] * \
                (self.window_size * 2 - len(context_idxs)) + context_idxs
        return torch.tensor(context_idxs), torch.tensor(target_idx)


def generate_cbow_dataset_cache():
    min_freq = VOCAB_CONFIG.min_freq
    max_freq = VOCAB_CONFIG.max_freq
    window_size = VOCAB_CONFIG.window_size
    cache_path = CACHE_CONFIG.val_cbow_dataset_cache_path

    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    cache_path = get_train_dataset_cache_path(min_freq, max_freq, window_size)
    cbow_dataset = CBOWDataset(dataset, vocab, window_size, cache_path)
    print(len(cbow_dataset))
    print(cbow_dataset[0])


def check_cbow_dataset_cache():
    files = []
    vocab = Vocab(VOCAB_CONFIG)
    for file in os.listdir('./cache/'):
        if file.endswith('.pkl') and file.startswith('train_dataset_'):
            files.append(file)

    for file in tqdm(files, desc="检查 CBOW 数据集缓存"):
        cache_path = os.path.join('./cache/', file)
        filename = cache_path.split('.')[-2]
        window_size = int(filename.split('_')[-1])
        cbow_dataset = CBOWDataset(
            None, vocab, window_size, cache_path)  # type: ignore
        for i in tqdm(range(len(cbow_dataset)), desc=f"检查 CBOW 数据集缓存 {filename}"):
            context, target = cbow_dataset[i]
            if context.size(0) != 2 * window_size:
                print(f"CBOW 数据集缓存 {file} 第 {i} 个样本的上下文窗口大小不一致")
            if target is None or not target:
                print(f"CBOW 数据集缓存 {file} 第 {i} 个样本的 target 为 None")


if __name__ == "__main__":
    # generate_cbow_dataset_cache()
    check_cbow_dataset_cache()
