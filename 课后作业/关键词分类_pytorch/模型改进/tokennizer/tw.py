import re
import nltk
from nltk.corpus import stopwords
import jieba
from .utils import get_stop_words, get_abs_path


def tokenize_tw(text: str) -> list[str]:
    stop_word_list = set(stopwords.words(
        'english') + stopwords.words('chinese') +
        get_stop_words('./stopwords/common.txt') + ['220v', '110v'])

    """
    + 符号在 keyword 中表示语义或者连接关系，需要替换为空格, 例如 sharp+microwave -> sharp microwave
    + 符号在 keyword 表示品牌名称，移除后不影响语义，例如 xiaomi x10+ -> xiaomi x10
    """
    keyword = re.sub('[+]', ' ', text)

    """
    & 符号在 keyword 中表示品牌名称，需要替换为 _， 避免被粉刺, 例如 charles & keith singapore -> charles_keith singapore
    """
    keyword = re.sub(r'(?<=\w)\s*&\s*(?=\w)', '_', keyword)

    """
    移除年份表示，例如 e-belia 2022 -> e-belia
    """
    keyword = re.sub(r'\b\d{4}\b', '', keyword)

    """
    xxx's 符号在 keyword 中表示 "什么什么的"，需要替换为空格, 例如 swisse men's vitality -> swisse men vitality
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    keyword = re.sub(r'(?<=\w)\'s(?=\b)', ' ', keyword)

    keyword = re.sub(r'[;\'"?!]$', ' ', keyword)

    def parse_num(match):
        num = re.sub(r'(g|G|T|t)', '', match.group(1))
        if num.isdigit() and int(num) % 2 == 0:
            return ' '
        else:
            return match.group()
    """
    存储单位 32g内存条 中 32g 是容量修饰词，内存是实体词, 无需保留单位
    """
    keyword = re.sub(
        r'\b(\d+(?:g|G|T|t))(?=[\u4e00-\u9fa5]|\s)', parse_num, keyword)

    jieba.load_userdict(get_abs_path('./userdict/tw.txt'))
    token_list = jieba.lcut(keyword)
    new_token_list = []
    for token in token_list:
        strip_token: str = token.strip()
        if strip_token and strip_token not in stop_word_list:
            if not strip_token.isascii() or len(strip_token) > 1:
                new_token_list.append(strip_token)

    return new_token_list
