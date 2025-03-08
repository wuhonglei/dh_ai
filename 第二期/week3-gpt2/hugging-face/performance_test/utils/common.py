import torch
from typing import Literal, List
import json


def get_device() -> Literal["mps", "cuda", "cpu"]:
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_text_token_len(text: List[str]) -> List[int]:
    return [len(t.split()) for t in text]


def get_max_token_len(prompt_len: List[int], story_len: List[int], min_len: int = 50, max_len: int = 900) -> int:
    avg_prompt_len = sum(prompt_len) / len(prompt_len)
    avg_story_len = sum(story_len) / len(story_len)
    return int(min(max(avg_prompt_len+avg_story_len, min_len), max_len))


def write_to_file(file_path: str, content: dict):
    # 追加写入
    with open(file_path, 'a') as f:
        f.write(json.dumps(content) + '\n')
