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
