import torch
import pickle
from torchtext.datasets import UDPOS
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, Vocab


class POSTagDataset(Dataset):
    def __init__(self, data: list[tuple[list[str], list[str], list[str]]]):
        self.examples: list[tuple[list[str], list[str]]] = list()  # 保存词性标注的数据
        for words, pos_tags, _ in data:
            lower_words = [word.lower() for word in words]
            lower_pos_tags = [pos_tag.lower() for pos_tag in pos_tags]
            self.examples.append((lower_words, lower_pos_tags))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def build_vocab(data):
    def yield_tokens(data_iter, index: int):
        for item in data_iter:
            yield item[index]

    special = ['<unk>', '<pad>']
    text_vocab = build_vocab_from_iterator(
        yield_tokens(data, 0), min_freq=2, specials=special)
    pos_vocab = build_vocab_from_iterator(
        yield_tokens(data, 1), min_freq=2, specials=special)

    # 将 '<unk>' 设置为默认索引
    text_vocab.set_default_index(text_vocab['<unk>'])
    pos_vocab.set_default_index(pos_vocab['<unk>'])

    # 返回文本词汇表和词性词汇表
    return text_vocab, pos_vocab


def collate_fn(batch: list[tuple[list[str], list[str]]], text_vocab: Vocab, pos_vocab: Vocab):
    text: list[torch.Tensor] = []
    pos: list[torch.Tensor] = []
    for text_sample, pos_sample in batch:
        text_tokens = [text_vocab[text_tag] for text_tag in text_sample]
        pos_tokens = [pos_vocab[pos_tag] for pos_tag in pos_sample]
        text.append(torch.LongTensor(text_tokens))
        pos.append(torch.LongTensor(pos_tokens))

    text = pad_sequence(
        text, padding_value=text_vocab['<pad>'], batch_first=True)  # type: ignore
    pos = pad_sequence(
        pos, padding_value=pos_vocab['<pad>'], batch_first=True)  # type: ignore
    return text, pos


if __name__ == '__main__':
    train_data, valid_data, test_data = UDPOS()
    dataset = POSTagDataset(test_data)
    text_vocab, pos_vocab = build_vocab(dataset)

    # def collate(batch): return collate_fn(batch, text_vocab, pos_vocab)
    # dataloader = DataLoader(test_data, batch_size=2,
    #                         shuffle=True, collate_fn=collate)

    # for batch_index, (text, pos_tag) in enumerate(dataloader):
    #     print(f'batch: {batch_index}')
    #     print(f'text shape: {text.shape}')
    #     print(f'pos_tag shape: {pos_tag.shape}')
    #     print('-------------------')
    #     if batch_index == 0:
    #         break
