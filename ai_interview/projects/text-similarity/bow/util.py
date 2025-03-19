import os
import readline
from config import DATA_CONFIG, CACHE_CONFIG, MILVUS_CONFIG


def init_dir():
    dir_path = [
        MILVUS_CONFIG.db_name,
        DATA_CONFIG.vocab_path,
        CACHE_CONFIG.search_history_path,
    ]
    for path in dir_path:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)


def setup_readline():
    # 设置历史文件路径
    readline.parse_and_bind('tab: complete')
    # 设置历史记录长度
    readline.set_history_length(1000)
    try:
        # 尝试从历史文件加载
        readline.read_history_file(CACHE_CONFIG.search_history_path)
    except FileNotFoundError:
        pass


def get_input() -> str | None:
    try:
        context = input('请输入搜索内容: ')
        # 保存到历史文件
        readline.write_history_file(CACHE_CONFIG.search_history_path)
        return context.strip()
    except KeyboardInterrupt:
        print('\n已取消输入')
        return None
