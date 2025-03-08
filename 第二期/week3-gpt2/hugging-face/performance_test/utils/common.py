import torch
from typing import Literal
import json


def get_device() -> Literal["mps", "cuda", "cpu"]:
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_text_token_len(text: str) -> int:
    return len(text.split())


def get_max_token_len(prompt_len: int, story_len: int, min_len: int = 50, max_len: int = 900) -> int:
    return min(max(prompt_len+story_len, min_len), max_len)


def write_to_file(file_path: str, content: dict):
    # 追加写入
    with open(file_path, 'a') as f:
        f.write(json.dumps(content) + '\n')
