import os
import pickle


def save_cache(file_path: str, data):
    print('saving cache to', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_cache(file_path: str):
    print('loading cache from', file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def exists_cache(file_path: str):
    return os.path.exists(file_path)


def write_to_file(path: str, content: str):
    # 追加内容到文件
    with open(path, "a") as f:
        f.write(content)
        f.write("\n")
