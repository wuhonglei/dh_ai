# fmt: off
import torch
from torch.utils.data import Dataset
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
# fmt: on


class NewsDataset(Dataset):
    def __init__(self, text_list: list[str], labels: list[int]):
        self.text_token_list: list[list[str]] = []
        tokenizer = get_tokenizer("basic_english")
        for text in text_list:
            tokenized_text: list[str] = [token.lower()
                                         for token in tokenizer(text)]
            self.text_token_list.append(tokenized_text)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[list[str], int]:
        return self.text_token_list[index], self.labels[index]


def build_vocab(dataset: NewsDataset):
    special = ["<unk>", "<pad>"]
    text_iter = map(lambda x: x[0], dataset)
    text_vocab = build_vocab_from_iterator(
        text_iter, min_freq=2, specials=special)
    text_vocab.set_default_index(text_vocab["<unk>"])
    return text_vocab


def collate_batch(batch: list[tuple[list[str], int]], text_vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    text_list, label_list = [], []
    for sentence, label in batch:
        indices = [text_vocab[token] for token in sentence]
        text_list.append(torch.LongTensor(indices))
        label_list.append(label)

    padding_idx = text_vocab["<pad>"]
    text_padded = pad_sequence(
        text_list, batch_first=True, padding_value=padding_idx)
    return text_padded, torch.tensor(label_list, dtype=torch.long)
