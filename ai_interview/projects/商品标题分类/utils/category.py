import json


def load_category_list(path: str):
    with open(path, 'r') as f:
        return json.load(f)['data']['list']


def get_category_by_id(id: str, category_list: list[dict]) -> dict | None:
    for category in category_list:
        if str(category['id']) == str(id):
            return category
    return None


def get_leaf_level_category_list(category: dict) -> list[dict]:
    leaf_category_list = []
    if not category:
        return leaf_category_list

    if not category['children']:
        leaf_category_list.append(category)
    else:
        for child in category['children']:
            leaf_category_list.extend(get_leaf_level_category_list(child))

    return leaf_category_list
