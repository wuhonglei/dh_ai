import json


def load_category_list(path: str):
    with open(path, 'r') as f:
        return json.load(f)['data']['list']
