from typing import Sequence
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
import re
import nltk
import string
import jieba  # 中文分词
from pythainlp.tokenize import word_tokenize as th_word_tokenize  # 泰文分词
from pythainlp.corpus.common import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn.utils.rnn import pad_sequence


def get_stop_words(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return file.read().split()


def tokenize_sg(text: str) -> list[str]:
    stop_word_list = set(stopwords.words(
        'english') + get_stop_words('./stopwords_custom.txt'))

    """
    + 符号在 keyword 中表示语义或者连接关系，需要替换为空格, 例如 sharp+microwave -> sharp microwave
    + 符号在 keyword 表示品牌名称，移除后不影响语义，例如 xiaomi x10+ -> xiaomi x10
    """
    keyword = re.sub('[+]', ' ', text)

    """
    & 符号在 keyword 中表示品牌名称，需要替换为 _， 避免被粉刺, 例如 charles & keith singapore -> charles_keith singapore
    """
    keyword = re.sub(r'(?<=\w)\s*&\s*(?=\w)', '_', keyword)

    """
    / 符号在数字中间，仅用于表示尺寸，因此需要移除左右两边的内容, 例如 3/4 pants mens -> pants mens
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    # keyword = re.sub(r'\d+/\d+', '', keyword)

    token_list = nltk.word_tokenize(keyword)
    new_token_list = []
    for token in token_list:
        strip_token = token.strip()
        if token not in stop_word_list and len(strip_token) > 1:
            new_token_list.append(strip_token)

    return new_token_list


def tokenize_my(text: str) -> list[str]:
    stop_word_list = set(stopwords.words(
        'english') + stopwords.words(
        'chinese') + get_stop_words('./stopwords_custom.txt'))

    """
    + 符号在 keyword 中表示语义或者连接关系，需要替换为空格, 例如 sharp+microwave -> sharp microwave
    + 符号在 keyword 表示品牌名称，移除后不影响语义，例如 xiaomi x10+ -> xiaomi x10
    """
    keyword = re.sub('[+]', ' ', text)

    """
    & 符号在 keyword 中表示品牌名称，需要替换为 _， 避免被粉刺, 例如 charles & keith singapore -> charles_keith singapore
    """
    keyword = re.sub(r'(?<=\w)\s*&\s*(?=\w)', '_', keyword)

    """
    xxx's 符号在 keyword 中表示 "什么什么的"，需要替换为空格, 例如 swisse men's vitality -> swisse men vitality
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    keyword = re.sub(r'(?<=\w)\'s(?=\b)', ' ', keyword)

    """
    移除年份表示，例如 e-belia 2022 -> e-belia
    """
    keyword = re.sub(r'\b\d{4}\b', '', keyword)

    """
    移除结尾的标点符号, 例如 ;'"
    """
    keyword = re.sub(r'[;\'"]$', '', keyword)

    token_list = jieba.cut(keyword)
    new_token_list = []
    for token in token_list:
        strip_token = token.strip()
        if token not in stop_word_list and len(strip_token) > 1:
            new_token_list.append(strip_token)

    return new_token_list


def tokenize_th(text: str) -> list[str]:
    stop_word_list = set(list(thai_stopwords()) +
                         get_stop_words('./stopwords_custom.txt'))

    """
    + 符号在 keyword 中表示语义或者连接关系，需要替换为空格, 例如 sharp+microwave -> sharp microwave
    + 符号在 keyword 表示品牌名称，移除后不影响语义，例如 xiaomi x10+ -> xiaomi x10
    """
    keyword = re.sub('[+]', ' ', text)

    """
    & 符号在 keyword 中表示品牌名称，需要替换为 _， 避免被粉刺, 例如 charles & keith singapore -> charles_keith singapore
    """
    keyword = re.sub(r'(?<=\w)\s*&\s*(?=\w)', '_', keyword)

    """
    移除年份表示，例如 e-belia 2022 -> e-belia
    """
    keyword = re.sub(r'\b\d{4}\b', '', keyword)

    """
    移除结尾的标点符号, 例如 ;'"
    """
    keyword = re.sub(r'[;\'"]$', '', keyword)

    token_list = th_word_tokenize(keyword)
    new_token_list = []
    for token in token_list:
        strip_token = token.strip()
        if token not in stop_word_list and len(strip_token) > 1:
            new_token_list.append(strip_token)

    return new_token_list


def tokenize_tw(text: str) -> list[str]:
    stop_word_list = set(stopwords.words(
        'english') + stopwords.words('chinese') +
        get_stop_words('./stopwords_custom.txt'))

    """
    + 符号在 keyword 中表示语义或者连接关系，需要替换为空格, 例如 sharp+microwave -> sharp microwave
    + 符号在 keyword 表示品牌名称，移除后不影响语义，例如 xiaomi x10+ -> xiaomi x10
    """
    keyword = re.sub('[+]', ' ', text)

    """
    & 符号在 keyword 中表示品牌名称，需要替换为 _， 避免被粉刺, 例如 charles & keith singapore -> charles_keith singapore
    """
    keyword = re.sub(r'(?<=\w)\s*&\s*(?=\w)', '_', keyword)

    """
    移除年份表示，例如 e-belia 2022 -> e-belia
    """
    keyword = re.sub(r'\b\d{4}\b', '', keyword)

    """
    xxx's 符号在 keyword 中表示 "什么什么的"，需要替换为空格, 例如 swisse men's vitality -> swisse men vitality
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    keyword = re.sub(r'(?<=\w)\'s(?=\b)', ' ', keyword)

    keyword = re.sub(r'[;\'"?!]$', ' ', keyword)

    """
    英文单词和中文之间增加空格，例如 三星samsung -> 三星 samsung
    jieba 分词支持拆分中英文，例如 三星samsung -> ['三星', 'samsung']
    """
    # keyword = re.sub(
    #     r'([a-zA-Z\d]+)(?=[\u4e00-\u9fa5])|([\u4e00-\u9fa5]+)(?=[a-zA-Z\d])', r'\1\2 ', keyword)

    """
    电压标识符号和中文之间增加空格，例如 220v冰箱 -> 220v 冰箱
    jieba 分词支持拆分中英文，例如 220v冰箱 -> ['220v', '冰箱']
    """
    # keyword = re.sub(
    #     r'(220v|110v)(?=[\u4e00-\u9fa5]+)|([\u4e00-\u9fa5]+)(?=220v|110v)', r'\1\2 ', keyword, flags=re.IGNORECASE)

    def parse_num(match):
        num = re.sub(r'(g|G|T|t)', '', match.group(1))
        if num.isdigit() and int(num) % 2 == 0:
            return match.group(1) + ' '
        else:
            return match.group()
    """
    数字和中文之间增加空格，例如 2g内存 -> 2g 内存
    jieba 分词支持拆分中英文，例如 2g内存 -> ['2g', '内存']
    """
    # keyword = re.sub(
    #     r'\b(\d+(?:g|G|T|t))(?=[\u4e00-\u9fa5]+)', parse_num, keyword)

    jieba.load_userdict('./userdict/tw.txt')
    token_list = jieba.lcut(keyword)
    new_token_list = []
    for token in token_list:
        strip_token: str = token.strip()
        if strip_token and strip_token not in stop_word_list:
            if not strip_token.isascii() or len(strip_token) > 1:
                new_token_list.append(strip_token)

    return new_token_list


token_dict = {
    'SG': tokenize_sg,
    'MY': tokenize_my,
    'TH': tokenize_th,
    'TW': tokenize_tw,
}


class KeywordCategoriesDataset(Dataset):
    def __init__(self, keywords: list[str], labels: list[str], country: str) -> None:
        unique_labels = list(set(labels))
        self.label2index = self.get_label_to_index(unique_labels)
        self.index2label = self.get_index_to_label(unique_labels)
        self.data = self.process_data(keywords, labels, country)

    def get_label_to_index(self, labels: Sequence[str]) -> dict[str, int]:
        label_to_index = {}
        for index, category in enumerate(labels):
            label_to_index[category] = index
        return label_to_index

    def get_index_to_label(self, labels: Sequence[str]) -> dict[int, str]:
        index_to_label = {}
        for index, category in enumerate(labels):
            index_to_label[index] = category
        return index_to_label

    def process_data(self, keywords: list[str], labels: list[str], country: str) -> list[tuple[list[str], str]]:
        # 遍历 dataframe
        data_list = []
        for index, keyword in enumerate(keywords):
            # 通过 index 获取 dataframe 的行
            category = labels[index]
            if not isinstance(keyword, str) or not isinstance(category, str):
                continue

            token_list = token_dict.get(country, tokenize_sg)(keyword)
            if token_list:
                data_list.append((token_list, self.label2index[category]))

        return data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[str], str]:
        return self.data[idx]

# 自定义分词器，仅按空格分割


def build_vocab(dataset: KeywordCategoriesDataset):
    # 使用自定义分词器
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(), token_pattern=None)  # 默认按照空白字符进行分词

    # 生成词汇表
    documents = [' '.join(x[0]) for x in dataset]
    vectorizer.fit(documents)

    # 获取现有的词汇表
    vocab = vectorizer.vocabulary_

    # 手动添加特殊标记
    special_tokens = {"<unk>": len(vocab), "<pad>": len(vocab) + 1}

    # 将特殊标记添加到词汇表中
    vocab.update(special_tokens)
    return vocab


def collate_batch(batch, vocab: dict[str, int]):
    text_list = list()
    labels = list()
    # 每次读取一组数据
    for text, label in batch:
        text_tokens = [vocab[token] if token in vocab else vocab['<unk>']
                       for token in text]
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_list.append(text_tensor)
        labels.append(torch.tensor(label, dtype=torch.long))

    padding_idx = vocab["<pad>"]
    # 将batch填充为相同长度文本
    text_padded = pad_sequence(
        text_list, batch_first=True, padding_value=padding_idx)
    # print("text_padded:", text_padded.shape)
    labels_tensor = torch.stack(labels)

    # 返回文本和标签的张量形式，用于后续的模型训练
    return text_padded, labels_tensor


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_excel('./data/Keyword_Categorization.xlsx', sheet_name='TW')
    keywords = data['Keyword'].tolist()
    labels = data['Category'].tolist()
    dataset = KeywordCategoriesDataset(keywords, labels, 'TW')
    vocab = build_vocab(dataset)
    pass
