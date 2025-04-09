"""
随机森林
  """

import pandas as pd
import os
from tqdm import tqdm
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.config import clean_dataset_dir


def get_df():
    return [
        pd.read_csv(os.path.join(clean_dataset_dir, csv_name))
        for csv_name in ['train.csv', 'valid.csv', 'test.csv']
    ]


train_df, valid_df, test_df = get_df()
results = []


def train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, title_name: str, label_name: str, feature_method: str):
    start_time = time.time()
    # 特征提取
    if feature_method == 'CountVectorizer':
        vectorizer = CountVectorizer()
    elif feature_method == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer()

    train_df = train_df.dropna(subset=[title_name, label_name])
    test_df = test_df.dropna(subset=[title_name, label_name])

    X_train = vectorizer.fit_transform(train_df[title_name])
    y_train = train_df[label_name]

    X_test = vectorizer.transform(test_df[title_name])
    y_test = test_df[label_name]

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{feature_method} {title_name} 准确率: {score}")

    end_time = time.time()
    results.append({
        'feature_method': feature_method,
        'title_name': title_name,
        'label_name': label_name,
        'data_size': X_train.shape[0],
        'vocab_size': X_train.shape[1],
        'accuracy': score,
        'train_time': end_time - start_time
    })
    print(f"{feature_method} {title_name} 训练时间: {end_time - start_time} 秒")


label_name = 'level1_global_be_category_id'
title_names = ['spacy_tokenized_name', 'nltk_tokenized_name',
               'remove_prefix', 'remove_prefix_emoji', 'remove_prefix_emoji_symbol']
methods = ['CountVectorizer', 'TfidfVectorizer']

for feature_method in methods:
    for title_name in tqdm(title_names, desc=f'训练模型: {feature_method}', total=len(title_names) * len(methods)):
        train_random_forest(valid_df, test_df, title_name,
                            label_name, feature_method)

df = pd.DataFrame(results)
df.to_csv('./results/base_model/random_forest_results.csv', index=False)
