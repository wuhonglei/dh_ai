"""
使用 fasttext 预测叶子目录
"""

import os
import time
import fasttext
import pandas as pd
from typing import Any
from tqdm import tqdm
from collections import Counter


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


def get_bucket(train_txt_path: str, wordNgrams: int, min_count: int) -> int:
    """
    获取桶的数量, 桶的数量由 wordNgrams 唯一值的数量决定
    """
    counter = Counter()
    with open(train_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()[1:]
            if len(words) < wordNgrams:
                continue
            for i in range(len(words) - wordNgrams + 1):
                ngram = words[i:i+wordNgrams]
                counter.update(ngram)

    bucket_count = 0
    for word, freq in counter.most_common():
        if freq < min_count:
            break
        bucket_count += 1

    print(f'bucket_count: {bucket_count}')
    return bucket_count


def train_fasttext_model(train_data_path: str, train_args: dict):
    train_args['bucket'] = get_bucket(
        train_data_path, train_args['wordNgrams'], train_args['minCount'])
    model = fasttext.train_supervised(input=train_data_path, **train_args)
    return model


def main():
    columns = [
        'remove_spacy_stop_words',
        'spacy_tokenized_name', 'nltk_tokenized_name',
        'remove_prefix', 'remove_prefix_emoji',
        'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words',
        'remove_nltk_stop_words',
    ]
    train_txt = 'train.txt'
    test_txt = 'test.txt'
    train_args = {
        'epoch': 80,
        'lr': 0.1,
        'wordNgrams': 2,
        'minCount': 2,
        'dim': 100,
        'loss': 'softmax',
    }

    result = []
    for column in tqdm(columns, desc='训练 fasttext 模型'):
        train_data_path = os.path.join('data', column, train_txt)
        test_data_path = os.path.join('data', column, test_txt)

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
            'accuracy': precision,
        })

    result_df = pd.DataFrame(result)
    csv_path = '../../../../results/leaf_level/fasttext_model/joint/results_from_scratch.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    result_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
