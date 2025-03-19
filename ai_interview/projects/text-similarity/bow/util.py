import os
import readline
from config import CACHE_CONFIG, MILVUS_CONFIG, VOCAB_CONFIG
import time
import json
from type_definitions import CategoryItem
from typing import Any


def init_dir():
    dir_path = [
        MILVUS_CONFIG.db_name,
        VOCAB_CONFIG.vocab_path,
        CACHE_CONFIG.search_history_path,
    ]
    for path in dir_path:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)


def setup_readline():
    # 确保历史文件目录存在
    os.makedirs(os.path.dirname(
        CACHE_CONFIG.search_history_path), exist_ok=True)

    # 设置历史文件
    try:
        readline.read_history_file(CACHE_CONFIG.search_history_path)
    except FileNotFoundError:
        # 如果历史文件不存在，则创建一个空文件
        with open(CACHE_CONFIG.search_history_path, 'w') as f:
            pass

    # 设置历史长度限制，避免文件过大
    readline.set_history_length(1000)

    # 启用自动补全功能
    readline.parse_and_bind('tab: complete')


def get_input() -> str | None:
    try:
        # 确保输入缓冲区清空
        import sys
        sys.stdout.flush()

        # 直接读取输入
        context = input('请输入搜索内容(输入q退出): ')

        # 只有在有效输入时才写入历史文件
        if context and context.strip():
            readline.write_history_file(CACHE_CONFIG.search_history_path)

        return context.strip() if context else ""
    except KeyboardInterrupt:
        print('\n已取消输入')
        return None
    except EOFError:
        print('\n检测到EOF')
        return None


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间为: {execution_time} 秒")
        return result
    return wrapper


def load_txt_file(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_json_file(file_path: str) -> Any:
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        return json.load(f)


def write_json_file(file_path: str, data: Any):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
