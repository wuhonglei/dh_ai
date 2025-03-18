import readline
from config import BowConfig


def setup_readline():
    # 设置历史文件路径
    readline.parse_and_bind('tab: complete')
    # 设置历史记录长度
    readline.set_history_length(1000)
    try:
        # 尝试从历史文件加载
        readline.read_history_file(BowConfig.search_history_path)
    except FileNotFoundError:
        pass


def get_input() -> str | None:
    try:
        context = input('请输入搜索内容: ')
        # 保存到历史文件
        readline.write_history_file(BowConfig.search_history_path)
        return context.strip()
    except KeyboardInterrupt:
        print('\n已取消输入')
        return None
