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
            data = load_pickle_file(self.cache_path)
            if data:
                return data

        if self.dataset is None or self.vocab is None:
            raise ValueError("dataset 和 vocab 不能为 None")

        # 使用CPU核心数量的进程
        num_processes = cpu_count()

        # 准备数据
        sentences = [self.dataset[i] for i in range(len(self.dataset))]

        # 创建进程池并处理数据
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_sentence, sentences),
                total=len(sentences),
                desc="生成 CBOW 训练数据"
            ))

        # 合并所有结果
        data: list[tuple[list[int], int]] = []
        for result in results:
            data.extend(result)

        if self.cache_path:
            save_pickle_file(self.cache_path, data)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        context_idxs, target_idx = self.data[index]
        # 确保上下文窗口大小一致
        if len(context_idxs) < self.window_size * 2:
            context_idxs = [self.vocab.pad_idx] * \
                (self.window_size * 2 - len(context_idxs)) + context_idxs
        return torch.tensor(context_idxs), torch.tensor(target_idx)


if __name__ == "__main__":
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
