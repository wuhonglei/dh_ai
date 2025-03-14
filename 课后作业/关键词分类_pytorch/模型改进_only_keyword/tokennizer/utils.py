import os
import re
import regex


def get_current_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_abs_path(file_path: str) -> str:
    return os.path.join(get_current_dir(), file_path)


def get_stop_words(file_path: str) -> list[str]:
    with open(get_abs_path(file_path), "r") as file:
        return file.read().split()


def add_space_between_unit(text: str, units: list[str]) -> str:
    """
    在 text 的单位之间添加空格
    :param text: 输入文本
    :param units: 单位列表
    :return: 添加空格后的文本
    @example:
    >>> add_space_between_unit("1kg", "1 kg ")
    """
    for unit in units:
        # 先处理 单位左侧含有数字的情况
        pattern = r"(?<=\d)\s*" + f'({re.escape(unit)})' + r"(?=([^a-zA-Z]|$))"
        text = re.sub(pattern, r' \1', text, re.IGNORECASE)

        # 再处理 单位右侧含有名词的情况
        pattern = r"(?<=\b)" + f'({re.escape(unit)})' + r"(?=[^a-zA-Z])"
        text = re.sub(pattern, r'\1 ', text, re.IGNORECASE)

    return text


def is_chinese_text(text: str) -> bool:
    """
    判断 text 是否为中文文本
    """
    return bool(re.search(r'^[\u4e00-\u9fa5]+$', text))


def contain_chinese_text(text: str) -> bool:
    """
    判断 text 是否包含中文文本
    """
    return bool(re.search(r'[\u4e00-\u9fa5]+', text))


def is_one_ascii(text: str) -> bool:
    """
    判断 text 是否为单个 ascii 字符
    """
    return bool(re.match(r'^[a-zA-Z\d]$', text))


def is_number(text: str) -> bool:
    """
    判断 text 是否为数字
    """
    return bool(re.match(r'^\d+$', text))


def is_float(text: str) -> bool:
    """
    判断 text 是否为浮点数
    """
    return bool(re.match(r'^\d+\.\d+$', text))


def split_graphemes(text: str) -> list[str]:
    """ Split text into graphemes """
    return regex.findall(r'\X', text)  # \X 表示完整的 Unicode Grapheme Cluster
