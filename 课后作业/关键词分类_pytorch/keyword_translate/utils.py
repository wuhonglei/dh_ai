from pandas import DataFrame
import json


def get_item_index(item, list):
    """ 获取列表中指定元素的索引 """
    indices = []
    for i, x in enumerate(list):
        if x == item:
            indices.append(i)

    return indices


def re_order_columns(df, new_index, category_name):
    if category_name not in df:
        df[category_name] = None

    columns: list[str] = df.columns.tolist()
    if columns[new_index] == category_name:
        return df

    removed_indices = get_item_index(category_name, columns)
    # 降序排列
    removed_indices.sort(reverse=True)
    temp_column = None
    for i in removed_indices:
        temp_column = df.iloc[:, i]
        columns.pop(i)
    columns.insert(new_index, category_name)
    if temp_column is not None:
        df[category_name] = temp_column

    return df[columns]


def save_csv(df: DataFrame, new_keyword_name: str, file_path: str):
    """ 保存数据到csv文件 """
    df = re_order_columns(df, 1, new_keyword_name)
    df.to_csv(file_path, index=False)


def load_translate_state():
    """ 从文件中加载翻译状态 """
    with open('./state.json', 'r') as f:
        return json.load(f)


def save_translate_state(state: dict):
    """ 保存翻译状态到文件 """
    with open('./state.json', 'w') as f:
        json.dump(state, f, indent=4)
