# from pythainlp.corpus.common import thai_stopwords
# from pythainlp.tokenize import word_tokenize as th_word_tokenize  # 泰文分词
import re
from .utils import get_stop_words, split_graphemes


def tokenize_th(text: str) -> list[str]:
    return split_graphemes(text)

    stop_word_list = set(list(thai_stopwords()) +
                         get_stop_words('./stopwords/common.txt'))

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
    移除结尾的标点符号, 例如 ;'"
    """
    keyword = re.sub(r'[;\'"]$', '', keyword)

    token_list = th_word_tokenize(keyword)
    new_token_list = []
    for token in token_list:
        strip_token = token.strip()
        if strip_token and strip_token not in stop_word_list:
            if not strip_token.isascii() or len(strip_token) > 1:
                new_token_list.append(strip_token)

    return new_token_list
