import os
import pandas as pd
import json

from translate.api import translate
from torch.utils.data import DataLoader
from dataset import KeywordDataset
from utils import re_order_columns, save_csv, load_translate_state, save_translate_state

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
    'sg',
    'my',
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
translate_state = load_translate_state()


def main():
    for country in countries:
        csv_file = f'./data/translated_csv/{country}.csv'
        src_language = language_code[country]
        df = pd.read_csv(csv_file)

        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        current_state = translate_state.get(country, {})
        if current_state.get('finish', False):
            print(f"Already translated {country}")
            continue

        new_keyword_name = f'{target_language}_Keyword'
        if src_language == target_language:
            save_csv(df, new_keyword_name,
                     f'./data/translated_csv/{country}.csv')
            print(f"Skip {country}")
            translate_state.update({country: {'finish': True}})
            continue

        dataset = KeywordDataset(df['Keyword'].tolist())
        batch_size = 100
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        last_index = current_state.get('error_batch_index', 0)
        for batch_index, (b_keyword, b_index) in enumerate(dataloader):
            # 跳过已经翻译过的
            if batch_index > last_index:
                continue

            print(
                f"Translating {country} batch {batch_index + 1}/{len(dataloader)}")

            b_keyword = list(b_keyword)
            b_index = b_index.tolist()
            translated: list[str] = translate(
                b_keyword, src_language, target_language)  # type: ignore

            if not translated:
                translate_state.update({
                    country: {'error_batch_index': batch_index,
                              'finish': False}
                })
                print(f"Failed to translate {country} batch {batch_index + 1}")
                continue

            for i, t in zip(b_index, translated):
                df.loc[i, new_keyword_name] = t

        save_csv(df, new_keyword_name,
                 f'./data/translated_csv/{country}.csv')
        translate_state.update({
            country: {
                'finish': True}
        })
        print(f"Translated {country}")


if __name__ == '__main__':
    main()
    save_translate_state(translate_state)
