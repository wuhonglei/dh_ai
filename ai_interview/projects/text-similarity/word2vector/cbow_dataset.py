from torch.utils.data import Dataset
from dataset import NewsDatasetCsv
from vocab import Vocab
from utils.common import load_pickle_file, save_pickle_file, timer_decorator
from tqdm import tqdm
from config import DATASET_CONFIG, CACHE_CONFIG, VOCAB_CONFIG
import torch


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

    @timer_decorator
    def load_data(self):
        if self.cache_path:
            data = load_pickle_file(self.cache_path)
            if data:
                return data

        if self.dataset is None or self.vocab is None:
            raise ValueError("dataset 和 vocab 不能为 None")

        data = []
        for i in tqdm(range(len(self.dataset)), desc="生成 CBOW 训练数据"):
            sentence = self.dataset[i]['content']
            words = self.vocab.tokenize(sentence)
            indices = self.vocab.batch_encoder(words)
            if len(indices) <= self.window_size:
                continue

            for i, target_index in enumerate(indices):
                context = indices[max(0, i - self.window_size):i] + \
                    indices[i + 1:i + self.window_size + 1]
                if context:
                    data.append((context, target_index))

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
    window_size = 8
    cache_path = 'cache/val_cbow_5.pkl'

    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    cbow_dataset = CBOWDataset(dataset, vocab, window_size, cache_path)
    print(len(cbow_dataset))
    print(cbow_dataset[0])
