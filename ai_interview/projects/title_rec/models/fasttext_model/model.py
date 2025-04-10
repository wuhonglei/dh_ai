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


def save_fasttext_model_as_vec(model, save_path: str) -> None:
    """
    将 fasttext 模型保存为 .vec 格式

    Args:
        model: fasttext模型
        save_path: 保存路径
    """
    # 获取词向量
    words = model.get_words()
    with open(save_path, 'w', encoding='utf-8') as f:
        # 写入词向量维度信息
        f.write(f"{len(words)} {model.get_dimension()}\n")
        # 写入每个词的向量
        for word in words:
            vector = model.get_word_vector(word)
            vector_str = ' '.join(map(str, vector))
            f.write(f"{word} {vector_str}\n")


def main():
    columns = [
        'remove_spacy_stop_words',
        'spacy_tokenized_name', 'nltk_tokenized_name',
        'remove_prefix', 'remove_prefix_emoji',
        'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words',
        'remove_nltk_stop_words',
    ]
    train_txt = 'valid.txt'
    test_txt = 'test.txt'
    train_args = {
        'epoch': 100,
        'lr': 0.1,
        'wordNgrams': 2,
        'minCount': 5,
        'dim': 100,
        'loss': 'softmax',
        'bucket': 35000,
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
            'accuracy': precision,
        })

    result_df = pd.DataFrame(result)
    result_df.to_csv(
        '../../results/fasttext_model/results_from_scratch.csv', index=False)


if __name__ == '__main__':
    main()
