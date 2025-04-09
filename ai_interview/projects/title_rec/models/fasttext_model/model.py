"""
使用 fasttext 进行商品标题分类
"""

import os
import time
import fasttext
import pandas as pd
from typing import Any
from tqdm import tqdm


def train_fasttext_model(train_data_path: str, train_args: dict):
    # 在训练时只使用训练集中的词汇
    train_args['minCount'] = 1  # 确保所有词汇都被包含
    train_args['minn'] = 0  # 禁用子词
    train_args['maxn'] = 0  # 禁用子词
    model = fasttext.train_supervised(input=train_data_path, **train_args)
    return model


def test_fasttext_model(model, test_data_path: str) -> float:
    score = model.test(test_data_path)
    return score[1]


def get_model_size(model) -> int:
    """
    计算模型大小（以MB为单位）

    Args:
        model: fasttext模型

    Returns:
        int: 模型大小（MB）
    """
    save_path = 'model.bin'
    model.save_model(save_path)
    size = os.path.getsize(save_path)  # 单位 字节
    os.remove(save_path)
    return size // (1024 * 1024)  # 单位 兆


def main():
    columns = [
        # 'spacy_tokenized_name', 'nltk_tokenized_name',
        # 'remove_prefix', 'remove_prefix_emoji',
        # 'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words',
        'remove_nltk_stop_words',
        'remove_spacy_stop_words'
    ]
    train_txt = 'valid.txt'
    test_txt = 'test.txt'
    train_args = {
        'epoch': 1,
        'lr': 0.1,
        'wordNgrams': 2,
        'minCount': 5,
        'dim': 300,
        'loss': 'softmax',
        'bucket': 200000,
        'pretrainedVectors': '',  # ./pretrained_vectors/crawl-300d-2M.vec
    }

    result = []
    for column in tqdm(columns, desc='训练 fasttext 模型'):
        train_data_path = os.path.join(
            'data', column, train_txt)
        test_data_path = os.path.join(
            'data', column, test_txt)

        start_time = time.time()
        model = train_fasttext_model(
            train_data_path, train_args)
        precision = test_fasttext_model(model, test_data_path)
        end_time = time.time()
        model_size = get_model_size(model)
        print(
            f'{column} 的 precision 为 {precision}, 训练时间: {end_time - start_time} 秒, 模型大小: {model_size} MB')
        result.append({
            'title_name': column,
            'train_txt': train_txt,
            'test_txt': test_txt,
            'model_size(MB)': model_size,
            'wordNgrams': train_args['wordNgrams'],
            'train_time(s)': end_time - start_time,
            'pretrainedVectors': os.path.basename(train_args['pretrainedVectors']),
            'accuracy': precision,
        })


if __name__ == '__main__':
    main()
