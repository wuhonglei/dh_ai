"""
处理 csv 文件, 生成 fasttext 训练数据
"""

import os
import pandas as pd
from tqdm import tqdm

from config import root_dir, dataset_dir, clean_dataset_dir, fasttext_dataset_dir


def save_txt(labels: list[str], titles: list[str], csv_path: str):
    with open(csv_path, 'w') as f:
        for label, title in zip(labels, titles):
            f.write(f'{label} {title}\n')


def generate_fasttext_dataset():
    label_name = 'level1_global_be_category_id'
    csv_names = ['test.csv', 'valid.csv', 'train.csv']
    columns = ['spacy_tokenized_name', 'nltk_tokenized_name',
               'remove_prefix', 'remove_prefix_emoji',
               'remove_prefix_emoji_symbol',
               'remove_prefix_emoji_symbol_stop_words',
               'remove_nltk_stop_words', 'remove_spacy_stop_words']
    for column in tqdm(columns, desc=f'生成 fasttext 训练数据'):
        output_dir = os.path.join(fasttext_dataset_dir, column)
        os.makedirs(output_dir, exist_ok=True)
        for csv_name in tqdm(csv_names, desc=f'生成 {column} 的 fasttext 训练数据'):
            csv_path = os.path.join(clean_dataset_dir, csv_name)
            df = pd.read_csv(csv_path)
            label = df[label_name].apply(lambda x: f'__label___{x}').tolist()
            title = df[column].tolist()
            csv_path = os.path.join(output_dir,
                                    f'{csv_name.split(".")[0]}.txt')
            save_txt(label, title, csv_path)
            # break
        # break


if __name__ == '__main__':
    generate_fasttext_dataset()
