import json
from type_define import CategoryItem, LeafCategoryItem


def load_category_list(path: str) -> list[CategoryItem]:
    with open(path, 'r') as f:
        return json.load(f)['data']['list']


def category_list_to_dict(category_list: list[CategoryItem]) -> dict[int, CategoryItem]:
    category_dict: dict[int, CategoryItem] = {}
    for category in category_list:
        category_dict[category['id']] = category
    return category_dict


def get_category_by_id(id: int, category_list: list[CategoryItem]) -> CategoryItem | None:
    for category in category_list:
        if category['id'] == id:
            return category
    return None


def get_leaf_level_category_list(category: CategoryItem) -> list[LeafCategoryItem]:
    leaf_category_list = []
    if not category:
        return leaf_category_list

    if not category['children']:
        leaf_category_list.append(category)
    else:
        for child in category['children']:
            leaf_category_list.extend(get_leaf_level_category_list(child))

    return leaf_category_list
