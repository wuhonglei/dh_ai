# fmt: off
import pickle
import torch
from torch.utils.data import Dataset
import spacy

from gensim.corpora import Dictionary
from torch.nn.utils.rnn import pad_sequence
# fmt: on


class NewsDataset(Dataset):
    def __init__(self, raw_sentence_list: list[str], labels: list[int], use_cache: bool = False):
        self.raw_sentence_list = raw_sentence_list
        cache_name = self.get_cache_name(raw_sentence_list)
        if use_cache and self.has_cache(cache_name):
            self.load_cache(cache_name)
            print(f"Loaded cache {cache_name}")
            return

        self.sentence_list: list[list[str]] = []
        # 获取spacy的英语分词器
        nlp = spacy.load('en_core_web_sm')
        total = len(raw_sentence_list)
        for i, sentence in enumerate(raw_sentence_list):
            print(f"Processing {i + 1}/{total}")
            word_list: list[str] = []
            for token in nlp(sentence.lower()):
                if token.is_stop or token.is_punct or token.is_space or len(token.text) < 2:
                    continue
                word_list.append(token.text)
            self.sentence_list.append(word_list)

        self.labels = labels
        self.save_cache(cache_name)

    def get_raw_sentence_length(self, raw_sentence_list: list[str]):
        return sum(len(sentence) for sentence in raw_sentence_list)

    def get_cache_name(self, raw_sentence_list: list[str]):
        total = self.get_raw_sentence_length(raw_sentence_list)
        return f"cache/dataset_{len(raw_sentence_list)}_{total}.pkl"

    def has_cache(self, cache_name: str):
        try:
            with open(cache_name, "rb") as f:
                return True
        except FileNotFoundError:
            return False

    def load_cache(self, cache_name: str):
        with open(cache_name, "rb") as f:
            self.sentence_list, self.labels = pickle.load(f)

    def save_cache(self, cache_name: str):
        with open(cache_name, "wb") as f:
            pickle.dump((self.sentence_list, self.labels), f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[list[str], int]:
        return self.sentence_list[index], self.labels[index]

    def get_sentence_in_vocab(self, vocab: Dictionary):
        sentences: list[list[str]] = []
        for sentence in self.sentence_list:
            new_sentence = []
            for word in sentence:
                if word in vocab.token2id:
                    new_sentence.append(word)
            if new_sentence:
                sentences.append(new_sentence)
        return sentences


def build_vocab(dataset: NewsDataset, special_tokens: list[str]):
    sentences_iter = map(lambda x: x[0], dataset)

    # 创建 Dictionary 对象
    dct = Dictionary(sentences_iter)

    # 过滤极低和极高频词
    dct.filter_extremes(no_below=5, no_above=0.5, keep_n=7000)

    if special_tokens is None:
        return dct

    # 创建新的词汇表列表，先添加特殊字符
    new_vocab = special_tokens.copy()

    # 添加现有词汇
    for token in dct.token2id:
        new_vocab.append(token)

    # 创建新的 Dictionary 对象
    new_dct = Dictionary()
    new_dct.add_documents([new_vocab])

    return new_dct


def collate_batch(batch: list[tuple[list[str], int]], text_vocab: Dictionary) -> tuple[torch.Tensor, torch.Tensor]:
    text_list, label_list = [], []
    token2id = text_vocab.token2id
    for sentence, label in batch:
        indices = [token2id.get(
            token, token2id['<unk>']) for token in sentence if token in token2id]
        text_list.append(torch.LongTensor(indices))
        label_list.append(label)

    padding_idx = token2id["<pad>"]
    text_padded = pad_sequence(
        text_list, batch_first=True, padding_value=padding_idx)
    return text_padded, torch.tensor(label_list, dtype=torch.long)
