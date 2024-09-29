import torch
from torch.utils.data import Dataset


class TextReader(Dataset):
    def context_to_vector(self, context: list[str], word2idx: dict[str, int]):
        return [word2idx[word] for word in context]

    def stat_raw_text(self, raw_text_list: list[str]):
        vocab = set(raw_text_list)
        vocab_size = len(vocab)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return vocab, vocab_size, word2idx, idx2word

    def make_train_data(self, raw_text_list: list[str], window: int):
        data: list[tuple[list[str], str]] = []
        start_index = window
        for i in range(start_index, len(raw_text_list) - window):
            context = raw_text_list[i - window:i] + \
                raw_text_list[i + 1:i + window + 1]
            target = raw_text_list[i]
            data.append((context, target))
        return data

    def __init__(self, raw_text_list: list[str], window: int):
        self.raw_text_list = raw_text_list
        self.window = window
        self.vocab, self.vocab_size, self.word2idx, self.idx2word = self.stat_raw_text(
            raw_text_list)

        self.raw_data = self.make_train_data(raw_text_list, window)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        context, target_tensor = self.raw_data[idx]
        context_tensor = torch.tensor(self.context_to_vector(
            context, self.word2idx), dtype=torch.long)
        target_tensor = torch.tensor(
            self.word2idx[target_tensor], dtype=torch.long)
        return context_tensor, target_tensor
