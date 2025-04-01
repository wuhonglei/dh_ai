import os
import pickle
import torch
import json


def save_cache(file_path: str, data):
    print('saving cache to', file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_cache(file_path: str):
    print('loading cache from', file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def exists_cache(file_path: str):
    return os.path.exists(file_path)


def get_device():
    """支持 mac gpu、nvidia gpu、cpu"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(file_path: str, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def create_dir(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path
