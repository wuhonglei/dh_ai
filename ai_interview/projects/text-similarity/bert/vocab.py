import jieba
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Literal
from collections import Counter, defaultdict
from dataset import NewsDatasetCsv
from utils.common import load_txt_file, load_json_file, timer_decorator, write_json_file, word_idf
import re
from transformers import BertTokenizer, BatchEncoding


from config import DATASET_CONFIG, VOCAB_CONFIG, VocabConfig

PATTERN_PADDING = Literal["max_length", "longest", "do_not_pad"]


class Vocab:
    def __init__(self, bert_name: str, max_length: int = 512):
        self.bert_name = bert_name
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            bert_name)
        self.max_length = max_length

    def tokenize(self, text: str, padding: PATTERN_PADDING = "max_length") -> BatchEncoding:
        return self.tokenizer(text, return_tensors="pt", padding=padding, truncation=True, max_length=self.max_length)

    def batch_encoder(self, texts: List[str], padding: PATTERN_PADDING = "max_length") -> BatchEncoding:
        return self.tokenizer(texts, return_tensors="pt", padding=padding, truncation=True, max_length=self.max_length)
