import os
import logging
import json
from datetime import datetime
from typing import Union, Any
from dataclasses import dataclass


def set_logger():
    current_date = datetime.now().strftime("%Y-%m-%d")
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
        handlers=[
            logging.FileHandler(f"logs/{current_date}.log"),  # 输出到文件
            logging.StreamHandler(),  # 输出到控制台
        ],
    )


def read_js_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def convert_to_dict(obj):
    if hasattr(obj, '__dict__'):
        return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    else:
        return obj


def read_json(file_path: str) -> Any:
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            return []


def write_json(file_path: str, data: Any):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_nd_json(file_path: str) -> list[dict]:
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line.strip()))
    return result


def write_nd_json(file_path: str, data: Union[list[dict], dict]):
    if isinstance(data, dict):
        data = [data]
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def hit_cache(url: str, json_list: list[dict]) -> bool:
    for item in json_list:
        if item["url"] == url:
            return True
    return False


@dataclass
class Test:
    url: str
    json_list: list[dict]

    def __getitem__(self, item: str) -> str:
        return getattr(self, item)
