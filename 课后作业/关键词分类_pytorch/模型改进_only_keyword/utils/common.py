import os
import pickle
import hashlib
import json


def save_cache(file_path: str, data, mode='wb'):
    print('saving cache to', file_path)
    with open(file_path, mode) as f:
        if mode == 'w':
            f.write(data)
        else:
            pickle.dump(data, f)


def save_json(file_path: str, data):
    print('saving json to', file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str):
    print('loading json from', file_path)
    with open(file_path, 'r') as f:
        return json.load(f)


def load_cache(file_path: str, mode='rb'):
    print('loading cache from', file_path)
    with open(file_path, mode) as f:
        if mode == 'r':
            return f.read()
        else:
            return pickle.load(f)


def exists_cache(file_path: str):
    return os.path.exists(file_path)


def get_file_state(file_path: str):
    return os.stat(file_path)


def write_to_file(path: str, content: str):
    # 追加内容到文件
    with open(path, "a") as f:
        f.write(content)
        f.write("\n")


def calculate_md5(string):
    md5 = hashlib.md5()
    md5.update(string.encode('utf - 8'))
    return md5.hexdigest()
