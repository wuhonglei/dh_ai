import os
import pandas as pd
import re


def find_first_category(cat_str: str) -> list[str]:
    """
    寻找第一个category
    """
    match = re.search(r'\[([^\[\]]*?)\]', cat_str)
    if not match:
        return []

    category_name = []
    for cat in re.findall(r'(?:\d+):(.+?)(?=,\s+\d{2,}|$)', match.group(1)):
        category_name.append(cat)

    return category_name


def origin_csv_rename():
    df = pd.read_csv('data/fe_category_20241105-150816.csv')
    df = df.dropna().drop_duplicates(
        subset=['name']).reset_index(drop=True)  # type: ignore
    data_list = []
    # 遍历所有行
    for i in range(len(df)):
        # 获取每行的item_id
        item_id = df.loc[i, 'item_id']
        # 商品名称
        name = df.loc[i, 'name']
        # 获取每行的category
        category: str = df.loc[i, 'fe_display_categories']  # type: ignore
        # 获取每行的category_list
        category_list = find_first_category(category)
        # 获取每行的category_list长度
        data_list.append([item_id, name] + category_list[0:3])

    new_df = pd.DataFrame(
        columns=['item_id', 'Keyword', 'fe_category_1', 'fe_category_2', 'fe_category_3'], data=data_list)

    new_df.to_csv('data/keyword_category.csv', index=False)


def keyword_len():
    df = pd.read_csv('data/keyword_category.csv')
    keyword_len = []
    for keyword in df['Keyword']:
        keyword_len.append(len(keyword.split()))

    new_df = pd.DataFrame(columns=['name_len'], data=keyword_len)
    new_df.to_csv('./data/name_len.csv', index=False)


if __name__ == '__main__':
    # origin_csv_rename()
    keyword_len()
    pass
