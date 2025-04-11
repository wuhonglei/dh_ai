"""
使用 fasttext 先预测一级目录，然后预测一级目录下的叶子目录
"""

import os
import time
import fasttext
import json
import numpy as np
import pandas as pd
from typing import Any
from tqdm import tqdm
from collections import Counter
from config import columns, train_txt, test_txt, train_args, level1_names, final_test_data_path


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


def get_txt_length(txt_path: str) -> int:
    with open(txt_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def predict_cascade(top_level_model, leaf_models: dict, txt_path: str) -> float:
    """
    预测级联模型
    """
    correct_count = 0
    total_count = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            text_list = line.strip().split()
            label = text_list[0].split('__label__')[1]
            title = ' '.join(text_list[1:])

            # 一级目录预测：获取 top-k 候选
            level1_preds, level1_probs = top_level_model.predict(title, k=10)
            level1_preds = [pred.replace('__label__', '')
                            for pred in level1_preds]
            level1_probs = np.log(level1_probs)  # 转换为对数概率

            # 扩展所有候选路径
            candidates = []
            for level1_pred, level1_prob in zip(level1_preds, level1_probs):
                leaf_model = leaf_models[level1_pred]
                leaf_preds, leaf_probs = leaf_model.predict(title, k=1)
                leaf_pred = leaf_preds[0].replace('__label__', '')
                leaf_prob = np.log(leaf_probs[0])

                # 计算联合得分
                joint_score = level1_prob + leaf_prob
                candidates.append({
                    'level1': level1_pred,
                    'leaf': leaf_pred,
                    'score': joint_score
                })

            best_candidate = max(candidates, key=lambda x: x['score'])
            leaf_pred = best_candidate['leaf']
            if label == leaf_pred:
                correct_count += 1
            total_count += 1

    return correct_count / total_count


def main():
    result = []
    for column in tqdm(columns, desc='训练 fasttext 模型'):
        # 先训练一级目录
        top_level_train_data_path = os.path.join(
            'data/top_level', column, train_txt)
        top_level_test_data_path = os.path.join(
            'data/top_level', column, test_txt)

        start_time = time.time()
        top_level_model = train_fasttext_model(
            top_level_train_data_path, train_args)
        top_level_precision = test_fasttext_model(
            top_level_model, top_level_test_data_path)
        end_time = time.time()
        top_level_model_size = get_model_size(top_level_model)
        print(
            f'{column} 的 precision 为 {top_level_precision}, 训练时间: {end_time - start_time} 秒, 模型大小: {top_level_model_size} MB')

        item = {
            'column': column,
            'top_level_precision': top_level_precision,
            'top_level_model_size': top_level_model_size,
            'top_level_train_time': end_time - start_time,
            'test_length': get_txt_length(top_level_test_data_path),
            'leaf_models': []
        }

        leaf_models = {}
        for level1_name in level1_names:
            train_data_path = os.path.join(
                'data/leaf_level', column, level1_name, train_txt)
            test_data_path = os.path.join(
                'data/leaf_level', column, level1_name, test_txt)

            start_time = time.time()
            model = train_fasttext_model(
                train_data_path, train_args)
            precision = test_fasttext_model(model, test_data_path)
            end_time = time.time()
            model_size = get_model_size(model)
            print(
                f'{column}.{level1_name} 的 precision 为 {precision}, 训练时间: {end_time - start_time} 秒, 模型大小: {model_size} MB')

            leaf_models[level1_name] = model
            item['leaf_models'].append({
                'level1_name': level1_name,
                'precision': precision,
                'model_size': model_size,
                'train_time': end_time - start_time,
                'test_length': get_txt_length(test_data_path)
            })

        accuracy = predict_cascade(top_level_model, leaf_models,
                                   final_test_data_path.format(column=column))
        item['cascade_accuracy'] = accuracy
        print(f'{column} 的准确率为 {accuracy}')
        result.append(item)
        break

    json_path = './result.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
