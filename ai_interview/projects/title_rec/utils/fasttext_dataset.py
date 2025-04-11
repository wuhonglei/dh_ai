"""
处理 csv 文件, 生成 fasttext 训练数据
"""

import os
import pandas as pd
from tqdm import tqdm

from config import root_dir, dataset_dir, clean_dataset_dir, fasttext_dataset_dir, fasttext_leaf_level_joint_dataset_dir, fasttext_leaf_level_cascade_dataset_dir


def save_txt(labels: list[str], titles: list[str], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w') as f:
        for label, title in zip(labels, titles):
            f.write(f'{label} {title}\n')


csv_names = ['test.csv', 'valid.csv']
columns = ['spacy_tokenized_name', 'nltk_tokenized_name',
           'remove_prefix', 'remove_prefix_emoji',
           'remove_prefix_emoji_symbol',
           'remove_prefix_emoji_symbol_stop_words',
           'remove_nltk_stop_words', 'remove_spacy_stop_words']


def generate_fasttext_dataset(label_name: str, dataset_dir: str):
    for column in tqdm(columns, desc=f'生成 fasttext 训练数据'):
        output_dir = os.path.join(dataset_dir, column)
        for csv_name in tqdm(csv_names, desc=f'生成 {column} 的 fasttext 训练数据'):
            csv_path = os.path.join(clean_dataset_dir, csv_name)
            df = pd.read_csv(csv_path)
            label = df[label_name].apply(lambda x: f'__label__{x}').tolist()
            title = df[column].tolist()
            csv_path = os.path.join(output_dir,
                                    f'{csv_name.split(".")[0]}.txt')
            save_txt(label, title, csv_path)
            # break
        # break


def generate_fasttext_dataset_for_every_top_level(level1_name: str, leaf_level_name: str, dataset_dir: str):
    for column in tqdm(columns, desc=f'生成 fasttext 训练数据'):
        for csv_name in tqdm(csv_names, desc=f'生成 {column} 的 fasttext 训练数据'):
            csv_path = os.path.join(clean_dataset_dir, csv_name)
            df = pd.read_csv(csv_path)
            for level1, df_level1 in df.groupby(level1_name):
                label = df_level1[leaf_level_name].apply(
                    lambda x: f'__label__{x}').tolist()
                title = df_level1[column].tolist()
                csv_path = os.path.join(dataset_dir, column, str(level1),
                                        f'{csv_name.split(".")[0]}.txt')
                save_txt(label, title, csv_path)


if __name__ == '__main__':
    generate_fasttext_dataset(label_name='global_be_category_id',
                              dataset_dir=os.path.join(fasttext_leaf_level_joint_dataset_dir))

    # generate_fasttext_dataset_for_every_top_level(
    #     level1_name='level1_global_be_category_id',
    #     leaf_level_name='global_be_category_id',
    #     dataset_dir=os.path.join(
    #         fasttext_leaf_level_cascade_dataset_dir, 'leaf_level')
    # )
