import os
import pandas as pd

from translate.api import translate
from torch.utils.data import DataLoader
from dataset import KeywordDataset
from utils import re_order_columns

# 映射为 IETF BCP 47 language tag
language_code = {
    'sg': 'en',
    'my': 'ms',
    'th': 'th',
    'tw': 'zh-TW',
    'id': 'id',
    'vn': 'vi',
    'ph': 'fil',
    'br': 'pt-BR',
    'mx': 'es-MX',
    'co': 'es-CO',
    'cl': 'es-CL',
}

countries = [
    # 'sg',
    # 'my',
    'th',
    'tw',
    'id',
    'vn',
    'ph',
    'br',
    'mx',
    'co',
    'cl'
]
target_language = 'en'  # 翻译为英文

for country in countries:
    csv_file = f'./data/unique_csv/{country}.csv'
    src_language = language_code[country]
    df = pd.read_csv(csv_file)

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        continue

    new_keyword_name = f'{target_language}_Keyword'
    if src_language == target_language:
        df[new_keyword_name] = df['Keyword'].str.lower()
        df = re_order_columns(df, 1, new_keyword_name)
        df.to_csv(f'./data/translated_csv/{country}.csv', index=False)
        print(f"Skip {country}")
        continue

    dataset = KeywordDataset(df['Keyword'].tolist())
    batch_size = 100
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_index, (b_keyword, b_index) in enumerate(dataloader):
        print(
            f"Translating {country} batch {batch_index + 1}/{len(dataloader)}")

        b_keyword = list(b_keyword)
        b_index = b_index.tolist()
        translated: list[str] = translate(
            b_keyword, src_language, target_language)  # type: ignore

        for i, t in zip(b_index, translated):
            df.loc[i, new_keyword_name] = t

    df = re_order_columns(df, 1, new_keyword_name)
    df.to_csv(f'./data/translated_csv/{country}.csv', index=False)
    print(f"Translated {country}")
