from openai import OpenAI
from typing import List, Tuple, Dict, Any
import pandas as pd
from tqdm import tqdm
from utils.common import exists_cache, load_cache, save_cache

client = OpenAI()


def get_embeddings(texts: List[str]):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [
        data.embedding for data in response.data
    ]


def main():
    df = pd.read_csv('./data/csv/sg.csv')
    keyname = 'Keyword'
    category_name = 'Category'
    data = df.dropna(subset=[keyname, category_name]).drop_duplicates(
        subset=[keyname], keep='first').reset_index(drop=True)  # type: ignore

    # 每次读取 500 条数据
    batch_size = 500
    cache_name = 'cache/sg_embeddings.pkl'

    progress = tqdm(range(0, len(data), batch_size),
                    desc='Embedding', total=len(data) // batch_size)
    for i in progress:
        new_data_dict: Dict[str, Any] = load_cache(
            cache_name) if exists_cache(cache_name) else {
            'last_row_index': -1,  # 记录上次读取的行位置
            keyname: [],
            category_name: [],
            'embeddings': []
        }

        if i <= new_data_dict['last_row_index']:
            continue

        X = data[keyname][i:i+batch_size].to_list()
        y = data[category_name][i:i+batch_size].to_list()
        new_data_dict['last_row_index'] = i
        new_data_dict[keyname].extend(X)
        new_data_dict[category_name].extend(y)
        new_data_dict['embeddings'].extend(get_embeddings(X))
        save_cache(cache_name, new_data_dict)


if __name__ == '__main__':
    main()
