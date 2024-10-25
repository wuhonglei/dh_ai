"""
读取 csv 文件，去重 Keyword 列
"""

import os
import pandas as pd

info = [
    ('csv_name', 'origin_len', 'unique_len', 'category_len'),
]
for csv_name in os.listdir('./data/origin_csv'):
    if not csv_name.endswith('.csv'):
        continue

    df = pd.read_csv(f'./data/origin_csv/{csv_name}')
    df['Keyword'] = df['Keyword'].str.lower()
    new_df = df.drop_duplicates(subset=['Keyword'],)
    new_df.to_csv(f'./data/unique_csv/{csv_name}', index=False)
    info.append(
        (csv_name.split('.')[0], len(df), len(
            new_df), len(new_df['Category'].unique()))  # type: ignore
    )

info_df = pd.DataFrame(info[1:], columns=info[0])
info_df.to_csv('./data/info.csv', index=False)
